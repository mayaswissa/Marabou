#include "DQNAgent.h"

#include "Options.h"
#include "RandomGlobals.h"

#include <limits>

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
    ASSERT( _numActions == _numPlConstraints * _numPhases );
    static_assert( DQN_RELU_ACTIVE == DQN_RELU_NOT_FIXED + 1 );
    static_assert( DQN_RELU_INACTIVE == DQN_RELU_NOT_FIXED + 2 );
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
    DQN_LOG( "saved agent's networks" );
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
        DQN_LOG( "loaded trained agent networks" );
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
                           const bool isDemo )
{
    _replayedBuffer.pushActionEntry( action, previousState, numSplits, isDemo, done );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( _tStep == 0 && GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

torch::Tensor Agent::applyActionMask( const torch::Tensor &tensorState,
                                      torch::Tensor &QValues ) const
{
    const int64_t P = (int64_t)_numPhases;
    const int64_t C = (int64_t)_numPlConstraints;
    int64_t B = 1;

    torch::Tensor notFixed;
    if ( tensorState.dim() == 2 )
    {
        notFixed =
            tensorState.select( 1, (int64_t)DQN_RELU_NOT_FIXED_VALUE ).unsqueeze( 0 );
    }
    else
    {
        notFixed = tensorState.select( 2, (int64_t)DQN_RELU_NOT_FIXED_VALUE );
        B = notFixed.size( 0 );
    }

    auto legalRows = notFixed.gt( 0.5 ).to( torch::kBool ).to( QValues.device() );
    auto qBCP =
        ( QValues.dim() == 1 ? QValues.view( { 1, C * P } ) : QValues ).reshape( { B, C, P } );
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    (void)qBCP.masked_fill_( ( ~legalRows ).unsqueeze( 2 ), NEG_INF );
    (void)qBCP.select( 2, (int64_t)DQN_RELU_NOT_FIXED ).fill_( NEG_INF );
    QValues = qBCP.view( { B, C * P } );
    return legalRows.sum( 1 ).eq( 0 );
}


std::unique_ptr<Action> Agent::actBestAction( const State &state )
{
    torch::NoGradGuard ng;
    _qNetworkLocal.eval();
    const auto tensorState = state.toTensor().to( device );
    torch::Tensor QValues = _qNetworkLocal.forward( tensorState );
    auto termMask = applyActionMask( tensorState, QValues );

    if ( termMask.to( torch::kCPU ).item<bool>() )
    {
        std::cerr << "Error: no valid actions in agent's best action!" << std::endl;
        throw std::runtime_error( "no valid actions." );
    }
    unsigned actionIndex = QValues.argmax( 1 ).item<int>();
    _qNetworkLocal.train();
    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    ASSERT( phase == DQN_RELU_ACTIVE || phase == DQN_RELU_INACTIVE );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}

std::unique_ptr<Action> Agent::actRandomly( const State &state )
{
    const auto tensorState = state.toTensor();
    auto reluNotFixedColumn = tensorState.index(
        { torch::indexing::Slice(), static_cast<int64_t>( DQN_RELU_NOT_FIXED_VALUE ) } );
    auto validRandomMask = reluNotFixedColumn.to( torch::kFloat32 ).gt( 0.5f );
    const torch::Tensor validRandomIndices = validRandomMask.nonzero();

    const auto k = validRandomIndices.size( 0 );
    if ( k == 0 )
        return nullptr;

    int row = RandomGlobals::instance().randInt( 0, static_cast<int>( k ) - 1 );
    const unsigned actionConstraint = validRandomIndices.index( { row, 0 } ).item<int>();

    // pick a random phase
    const unsigned actionPhase =
        RandomGlobals::instance().randInt( DQN_RELU_ACTIVE, DQN_RELU_INACTIVE );

    const unsigned actionIndex = _actionSpace.getActionIndex( actionConstraint, actionPhase );
    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}


void Agent::learn()
{
    auto batch = _replayedBuffer.sample();
    if ( batch.indices.empty() )
        return;

    auto idxTensor = torch::tensor(
        std::vector<int64_t>( batch.indices.begin(), batch.indices.end() ), torch::kLong );

    const auto statesTensor =
        _replayedBuffer.getStates().index( { idxTensor } ).to( device ).to( torch::kFloat32 );
    const auto actionsTensor =
        _replayedBuffer.getActions().index( { idxTensor } ).to( device ).to( torch::kLong );
    const auto rewardsTensor =
        _replayedBuffer.getRewards().index( { idxTensor } ).to( device ).to( torch::kFloat32 );
    const auto nextStatesTensor =
        _replayedBuffer.getNextStates().index( { idxTensor } ).to( device ).to( torch::kFloat32 );
    auto doneTensor = _replayedBuffer.getDones()
                          .index( { idxTensor } )
                          .to( device )
                          .to( torch::kBool )
                          .view( { -1 } );

    auto all_q = _qNetworkLocal.forward( statesTensor );
    const auto QExpected = all_q.gather( 1, actionsTensor ).squeeze( -1 ).to( torch::kFloat32 );
    auto QTargets = rewardsTensor;

    // --- Double DQN ---
    {
        torch::NoGradGuard _ng;
        auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
        auto termMaskLocal = applyActionMask( nextStatesTensor, forwardLocalNet );
        auto bad = ( ( ~doneTensor ) & termMaskLocal );
        const auto localQValuesNextState = forwardLocalNet.argmax( 1, true );

        auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
        (void)applyActionMask( nextStatesTensor, forwardTargetNet );
        const auto targetQValuesNextState =
            forwardTargetNet.gather( 1, localQValuesNextState ).squeeze( 1 );

        auto notDone = ((~doneTensor) & (~termMaskLocal)).to(torch::kBool);
        auto gatedTarget = torch::where( notDone, targetQValuesNextState, torch::zeros_like(targetQValuesNextState) );
        QTargets = rewardsTensor + GAMMA * gatedTarget;
    }

    if ( torch::isnan( QTargets ).any().item<bool>() )
    {
        std::cerr << "Error: QTargets contains NaN values!" << std::endl;
        throw std::runtime_error( "NaN detected in QTargets." );
    }

    // --- TD loss ---
    auto td_errors = torch::smooth_l1_loss( QExpected, QTargets.detach(), torch::Reduction::None );
    auto weights = torch::tensor( batch.weights, statesTensor.options().dtype( torch::kFloat32 ) )
                       .to( device );
    weights = weights / weights.mean().clamp_min( 1e-8f );
    auto weightedTdLoss = ( td_errors * weights ).mean();

    // --- Margin loss (demonstration data) ---
    std::vector<int64_t> demo_mask_int( batch.isDemo.begin(), batch.isDemo.end() );
    if ( !torch::isfinite( all_q ).all().item<bool>() )
    {
        throw std::runtime_error( "Non-finite all_q in learn() (pre-mask)" );
    }

    auto all_q_masked = all_q.clone();
    auto termMask = applyActionMask( statesTensor, all_q_masked );

    auto demo_mask = torch::tensor( demo_mask_int, torch::TensorOptions().dtype( torch::kInt64 ) )
                         .to( device )
                         .to( torch::kFloat32 );
    auto valid_demo = ( demo_mask * ( 1.0f - termMask.to( torch::kFloat32 ) ) );
    auto flat_actions = actionsTensor.squeeze( -1 );

    auto q_demo = all_q_masked.gather( 1, flat_actions.unsqueeze( 1 ) ).squeeze( 1 );

    auto shifted = all_q_masked + _margin; // [B,A]
    (void)shifted.scatter_( 1,
                            flat_actions.unsqueeze( 1 ),
                            all_q_masked.gather( 1, flat_actions.unsqueeze( 1 ) ) );

    auto max_other = std::get<0>( shifted.max( 1 ) );
    auto raw_margin = torch::relu( max_other - q_demo );
    auto marginLoss = ( raw_margin * valid_demo ).sum() / valid_demo.sum().clamp_min( 1.0f );

    // --- Total loss ---
    const auto loss = weightedTdLoss + _lambdaSup * marginLoss;
    _lossVerbosity = ( _lossVerbosity + 1 ) % 100;
    if ( _lossVerbosity == 0 )
    {
        DQN_LOG( Stringf( "TD Loss : %.10f\n", weightedTdLoss.item<double>() ).ascii() );
        DQN_LOG( Stringf( "MARGIN Loss : %.10f\n", marginLoss.item<double>() ).ascii() );
        DQN_LOG( Stringf( "Loss : %.10f\n", loss.item<double>() ).ascii() );
    }

    // --- Backpropagation ---
    _optimizer.zero_grad( true );
    loss.backward();
    torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 1.0 );
    if ( !handleInvalidGradients() )
        _optimizer.step();
    softUpdate( _qNetworkLocal, _qNetworkTarget );

    auto abs_td =
        ( QTargets.detach() - QExpected.detach() ).abs().to( torch::kCPU ).contiguous(); // [B]
    auto acc = abs_td.accessor<float, 1>();
    for ( size_t i = 0; i < batch.indices.size(); ++i )
    {
        float base = batch.isDemo[i] ? static_cast<float>( _replayedBuffer.getEpsilonDemo() )
                                     : static_cast<float>( _replayedBuffer.getEpsilonAgent() );
        _replayedBuffer.updatePriority( batch.indices[i], acc[i] + base );
    }

    if ( GlobalConfiguration::DON_TRAINING_PHASE == 2 )
        _lambdaSup = std::fmax( 0.1f, _lambdaSup - _lambdaDecay );
}


void Agent::softUpdate( const QNetwork &localModel, QNetwork &targetModel )
{
    torch::NoGradGuard _;
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