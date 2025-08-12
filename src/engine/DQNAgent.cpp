#include "DQNAgent.h"

#include "Options.h"
#include "RandomGlobals.h"

#include <random>
#include <utility>

Agent::Agent( const unsigned numPlConstraints,
              const unsigned numPhases,
              const std::string &trainedAgentPath )
    : _actionSpace( ActionSpace( numPlConstraints, numPhases ) )
    , _numPlConstraints( numPlConstraints )
    , _numPhases( numPhases )
    , _numActions( _actionSpace.getNumActions() )
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _trainedAgentFilePath( trainedAgentPath )
    , _qNetworkLocal(
          QNetwork( _numPlConstraints, NUM_LOCAL_FEATURES, _numActions, NUM_GLOBAL_FEATURES ) )
    , _qNetworkTarget(
          QNetwork( _numPlConstraints, NUM_LOCAL_FEATURES, _numActions, NUM_GLOBAL_FEATURES ) )
    , _optimizer( _qNetworkLocal.parameters(),
                  torch::optim::AdamOptions( Options::get()->getFloat( Options::DQN_LR ) )
                      .weight_decay( Options::get()->getFloat( Options::DQN_WEIGHT_DECAY ) ) )
    , _scheduler( _optimizer, 4, 0.95 )
    , _replayedBuffer( ReplayBuffer( _numPlConstraints,
                                     Options::get()->getInt( Options::DQN_BUFFER_SIZE ),
                                     Options::get()->getInt( Options::DQN_BATCH_SIZE ) ) )
    , _lossVerbosity( 0 )
    , _lambdaSup( Options::get()->getFloat( Options::DQfD_LAMBDA_SUP ) )
    , _lambdaDecay( Options::get()->getFloat( Options::DQfD_LAMBDA_DECAY ) )
    , _margin( Options::get()->getFloat( Options::DQfD_MARGIN ) )
{
    _qNetworkLocal.to( device );
    _qNetworkTarget.to( device );
    _qNetworkLocal.to( torch::kFloat32 );
    _qNetworkTarget.to( torch::kFloat32 );
    // If a load path is provided, load the networks
    if ( !trainedAgentPath.empty() )
        loadNetworks();
}

void Agent::saveNetworks( const std::string &path ) const
{
    // Save local network
    {
        torch::serialize::OutputArchive local_archive;
        _qNetworkLocal.save( local_archive );
        local_archive.save_to( path + "_local.pth" );
    }

    // Save target network
    {
        torch::serialize::OutputArchive target_archive;
        _qNetworkTarget.save( target_archive );
        target_archive.save_to( path + "_target.pth" );
    }
    DQN_LOG( "saved agent's networks" )
}


void Agent::loadNetworks()
{
    try
    {
        // Load local network
        {
            torch::serialize::InputArchive local_archive;
            local_archive.load_from( _trainedAgentFilePath + "_local.pth" );
            _qNetworkLocal.load( local_archive );
        }

        // Load target network
        {
            torch::serialize::InputArchive target_archive;
            target_archive.load_from( _trainedAgentFilePath + "_target.pth" );
            _qNetworkTarget.load( target_archive );
        }
        DQN_LOG( "loaded trained agent networks" )
    }
    catch ( const torch::Error &e )
    {
        std::cerr << "Failed to load networks: " << e.what() << std::endl;
    }
}

bool Agent::handleInvalidGradients()
{
    bool invalid = false;
    for ( auto &group : _optimizer.param_groups() )
    {
        for ( auto &p : group.params() )
        {
            if ( p.grad().defined() && ( torch::isnan( p.grad() ).any().item<bool>() ||
                                         torch::isinf( p.grad() ).any().item<bool>() ) )
            {
                std::cerr << "Invalid gradient detected, resetting gradient..." << std::endl;
                p.grad().detach_();
                p.grad().zero_();
                invalid = true;
            }
        }
    }
    return invalid;
}


void Agent::handleDone( const State &currentState, const unsigned numSplits )
{
    // Insert all actions from actions buffer to the replay buffer and learn.
    _replayedBuffer.handleDone( currentState, numSplits );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

void Agent::stepAlternativeAction( const State &stateBeforeSplit,
                                   const unsigned numSplits,
                                   unsigned &numInconsistent )
{
    _replayedBuffer.applyNextAction( stateBeforeSplit, numSplits, numInconsistent );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( _tStep == 0 && GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

void Agent::stepFakeAction( const State &stateBeforeAction, const unsigned numSplitsBeforeAction )
{
    _replayedBuffer.pushFakeActionEntry( stateBeforeAction, numSplitsBeforeAction );
}

void Agent::stepNewAction( const State &previousState,
                           const Action &action,
                           const bool done,
                           const unsigned numSplits,
                           bool isDemo = false )
{
    _replayedBuffer.pushActionEntry( action, previousState, numSplits, isDemo, done );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( _tStep == 0 && GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

torch::Tensor Agent::maskQInPlace( const torch::Tensor &state, torch::Tensor &Q ) const
{
    constexpr float NEG = -1e9f;   // large finite to avoid inf math
    const auto opts = Q.options(); // match device + dtype

    // Bring state to Q's device/dtype
    auto s = state.to( opts );

    // Extract NOT_FIXED feature into [B,C]
    torch::Tensor notFixed;
    int64_t B = 1, C = (int64_t)_numPlConstraints, P = (int64_t)_numPhases;
    if ( s.dim() == 2 )
    {                                                                               // [C,F]
        notFixed = s.select( 1, (int64_t)DQN_RELU_NOT_FIXED_VALUE ).unsqueeze( 0 ); // [1,C]
    }
    else
    {                                                                // [B,C,F]
        notFixed = s.select( 2, (int64_t)DQN_RELU_NOT_FIXED_VALUE ); // [B,C]
        B = notFixed.size( 0 );
        C = notFixed.size( 1 );
    }

    // Legal rows = NOT_FIXED; terminals if none
    auto rowLegal = notFixed.gt( 0.5 );              // [B,C] bool
    auto terminalByMask = rowLegal.sum( 1 ).eq( 0 ); // [B]   bool

    // Legal phases = {ACTIVE, INACTIVE}
    auto phaseIdx =
        torch::arange( P, torch::TensorOptions().device( Q.device() ).dtype( torch::kLong ) )
            .view( { 1, 1, -1 } );
    auto phaseLegal = phaseIdx.ne( (int64_t)DQN_RELU_NOT_FIXED ); // [1,1,P] bool

    // legal3D[b,c,p] = rowLegal[b,c] && phaseLegal[p]
    auto legal3D = rowLegal.unsqueeze( 2 ) & phaseLegal; // [B,C,P]

    // Mask by ASSIGNMENT (no +/−inf arithmetic)
    if ( Q.dim() == 3 )
    { // [B,C,P]
        Q.masked_fill_( ~legal3D, NEG );
    }
    else
    { // [B,A] with A=C*P
        Q.masked_fill_( ( ~legal3D ).reshape( { B, C * P } ), NEG );
    }

    return terminalByMask;
}


static inline void sanitizeInPlace( torch::Tensor &T )
{
    auto bad = ~torch::isfinite( T );
    if ( bad.any().item<bool>() )
    {
        T.masked_fill_( bad, 0 ); // NaN/±inf → 0
        T.clamp_( -1e9f, 1e9f );  // optional but helps stability
    }
}

std::unique_ptr<Action> Agent::actBestAction( const State &state )
{
    _qNetworkLocal.eval();
    const auto tensorState = state.toTensor().to( device );
    torch::Tensor QValues = _qNetworkLocal.forward( tensorState );
    sanitizeInPlace( QValues );
    _qNetworkLocal.train();
    if ( maskQInPlace( tensorState, QValues ).item<bool>() )
        return nullptr;
    unsigned actionIndex = QValues.argmax( 1 ).item<int>();
    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}

std::unique_ptr<Action> Agent::actRandomly( const State &state )
{
    // State tensor is [C, F] on CPU
    const auto s = state.toTensor();

    // Read the NOT_FIXED feature flag column (robust to float dtype)
    const auto notFixedCol =
        s.index( { torch::indexing::Slice(), (int64_t)DQN_RELU_NOT_FIXED_VALUE } )
            .to( torch::kFloat32 );

    // Collect candidate constraint rows (NOT_FIXED == true)
    std::vector<unsigned> candidates;
    candidates.reserve( _numPlConstraints );
    for ( unsigned c = 0; c < _numPlConstraints; ++c )
    {
        if ( notFixedCol[c].item<float>() > 0.5f )
            candidates.push_back( c );
    }

    // No legal rows left → let caller treat as terminal
    if ( candidates.empty() )
        return nullptr;

    // Uniformly pick a candidate row
    const int pick = RandomGlobals::instance().randInt( 0, (int)candidates.size() - 1 );
    const unsigned constraint = candidates[(size_t)pick];

    // Uniformly pick a legal phase (NOT_FIXED is disallowed)
    const unsigned phase = ( RandomGlobals::instance().randInt( 0, 1 ) == 0 )
                             ? (unsigned)DQN_RELU_ACTIVE
                             : (unsigned)DQN_RELU_INACTIVE;

    // Build action
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}


void Agent::learn()
{
    auto batch = _replayedBuffer.sample();
    if ( batch.indices.empty() )
        return;

    // indices [B]
    std::vector<int64_t> idx64( batch.indices.begin(), batch.indices.end() );
    auto idxTensor = torch::tensor( idx64, torch::dtype( torch::kLong ) );

    // batch tensors
    auto statesTensor = _replayedBuffer.getStates().index( { idxTensor } ).to( device );
    auto actionsTensorB =
        _replayedBuffer.getActions().index( { idxTensor } ).to( device ).to( torch::kLong ); // [B]
                                                                                             // Long
    auto rewardsTensor = _replayedBuffer.getRewards()
                             .index( { idxTensor } )
                             .to( device )
                             .to( torch::kFloat32 ); // [B]
    auto nextStatesTensor = _replayedBuffer.getNextStates().index( { idxTensor } ).to( device );
    auto doneTensor = _replayedBuffer.getDones()
                          .index( { idxTensor } )
                          .to( device )
                          .to( torch::kBool )
                          .view( { -1 } ); // [B] bool

    // Q(s,a) with correct gather shape
    auto Qs = _qNetworkLocal.forward( statesTensor ); // [B, A]
    sanitizeInPlace( Qs );
    auto QExpected = Qs.gather( 1, actionsTensorB.view( { -1, 1 } ) ) // [B,1]
                         .squeeze( 1 )                                // [B]
                         .to( torch::kFloat32 );

    // ----- Double DQN targets -----
    // local: pick argmax_a' Q(s',a')
    auto next_local = _qNetworkLocal.forward( nextStatesTensor ); // [B, A]
    auto termMask = maskQInPlace( nextStatesTensor, next_local ); 
    auto next_actions = next_local.detach().argmax( 1 ).view( { -1, 1 } ); // [B,1] Long

    // target: evaluate those actions
    auto next_target = _qNetworkTarget.forward( nextStatesTensor ); // [B, A]
    sanitizeInPlace( next_target );
    maskQInPlace( nextStatesTensor, next_target );
    auto targetQValuesNextState =
        next_target.detach().gather( 1, next_actions ).squeeze( 1 ); // [B]

    // targets
    auto notDone = ( ( ~doneTensor ) & ( ~termMask ) ).to( torch::kFloat32 );
    auto QTargets = rewardsTensor + GAMMA * targetQValuesNextState * notDone; // [B]

    if ( torch::isnan( QTargets ).any().item<bool>() )
    {
        std::cerr << "Error: QTargets contains NaN values!" << std::endl;
        throw std::runtime_error( "NaN detected in QTargets." );
    }

    // ----- TD loss (PER-weighted Huber) -----
    auto td_errors =
        torch::smooth_l1_loss( QExpected, QTargets.detach(), torch::Reduction::None ); // [B]
    auto weights = torch::tensor( batch.weights, statesTensor.options().dtype( torch::kFloat32 ) )
                       .to( device );                     // [B]
    weights = weights / weights.mean().clamp_min( 1e-8 ); // normalize to keep scale stable
    auto weightedTdLoss = ( td_errors * weights ).mean(); // scalar

    // ----- Large-margin supervised loss (only on demo samples) -----
    std::vector<int64_t> demo_mask_int( batch.isDemo.begin(), batch.isDemo.end() );
    auto demo_mask =
        torch::tensor( demo_mask_int, torch::kLong ).to( device ).to( torch::kFloat32 ); // [B]

    torch::Tensor margin_loss = torch::zeros( {}, statesTensor.options().dtype( torch::kFloat32 ) );
    if ( demo_mask.sum().item<float>() > 0.0f )
    {
        auto all_q = _qNetworkLocal.forward( statesTensor ); // [B, A]
        sanitizeInPlace( all_q );
        maskQInPlace( statesTensor, all_q ); // mask illegal actions inside Q

        auto a = actionsTensorB.view( { -1 } ); // [B] Long
        auto one_hot = torch::nn::functional::one_hot( a, (int64_t)_numActions )
                           .to( device )
                           .to( torch::kFloat32 );                       // [B, A]
        auto non_selected = 1.0f - one_hot;                              // [B, A]
        auto shifted = all_q + non_selected * _margin;                   // [B, A]
        auto max_other = std::get<0>( shifted.max( 1 ) );                // [B]
        auto q_demo = all_q.gather( 1, a.unsqueeze( 1 ) ).squeeze( 1 );  // [B]
        auto raw_margin = torch::relu( max_other - q_demo ) * demo_mask; // [B]
        margin_loss = raw_margin.sum() / demo_mask.sum().clamp_min( 1.0f );
    }

    // total loss
    auto loss = weightedTdLoss + _lambdaSup * margin_loss;

    // ----- Logging (occasionally) -----
    _lossVerbosity = ( _lossVerbosity + 1 ) % 100;
    if ( _lossVerbosity == 0 )
    {
        DQN_LOG( Stringf( "TD Loss: %.6f  Margin: %.6f  Total: %.6f",
                          weightedTdLoss.item<double>(),
                          margin_loss.item<double>(),
                          loss.item<double>() )
                     .ascii() );
    }

    // ----- Backprop -----
    _optimizer.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 1.0 );
    if ( !handleInvalidGradients() )
        _optimizer.step();
    softUpdate( _qNetworkLocal, _qNetworkTarget );

    if ( GlobalConfiguration::DON_TRAINING_PHASE == 2 )
        _lambdaSup = std::fmax( 0.1f, _lambdaSup - _lambdaDecay );


    // ----- Update PER priorities: |TD error| + eps -----
    auto abs_td = torch::abs( QTargets.detach() - QExpected.detach() ).to( torch::kCPU ); // [B]
    auto acc = abs_td.accessor<float, 1>();
    for ( size_t i = 0; i < batch.indices.size(); ++i )
    {
        float base_eps = batch.isDemo[i] ? (float)_replayedBuffer.getEpsilonDemo()
                                         : (float)_replayedBuffer.getEpsilonAgent();
        float p = acc[i] + base_eps;
        if ( !std::isfinite( p ) )
            p = base_eps;
        _replayedBuffer.updatePriority( batch.indices[i], p );
    }
}


void Agent::softUpdate( const QNetwork &localModel, const QNetwork &targetModel )
{
    const auto localParams = localModel.getParameters();
    const auto targetParams = targetModel.getParameters();
    for ( size_t i = 0; i < localParams.size(); ++i )
    {
        targetParams[i].data().copy_( GlobalConfiguration::DQN_TAU * localParams[i].data() +
                                      ( 1.0 - GlobalConfiguration::DQN_TAU ) *
                                          targetParams[i].data() );
    }
}

torch::Device Agent::getDevice() const
{
    return device;
}

int Agent::getActionStackSize() const
{
    return _replayedBuffer.getActionStackSize();
}

int Agent::getReplayBufferSize() const
{
    return _replayedBuffer.getNumRevisitExperiences();
}

void Agent::schedulersStep()
{
    _scheduler.step();
}
