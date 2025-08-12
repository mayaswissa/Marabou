// DQNAgent.cpp — robust “best-of” variant (respects DQN_EXPLORATION_RATE)
#include "DQNAgent.h"

#include "Options.h"
#include "RandomGlobals.h"

#include <cmath>
#include <limits>
#include <random>
#include <utility>

// ───────────────────────── helpers ─────────────────────────
static inline float perStepDecayToReach( float start, float floor, int steps )
{
    if ( steps <= 0 || start <= 0.0f || floor <= 0.0f ) return 1.0f;
    floor = std::min( floor, start ); // never go up
    return std::pow( floor / start, 1.0f / static_cast<float>( steps ) );
}

static inline void sanitizeInPlace( torch::Tensor &T )
{
    auto bad = ~torch::isfinite( T );
    if ( bad.any().item<bool>() )
    {
        T.masked_fill_( bad, 0 );  // NaN/±inf → 0
        T.clamp_( -1e9f, 1e9f );   // tame extremes
    }
}

// ───────────────────────── ctor ─────────────────────────
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
    , _lambdaDecay( Options::get()->getFloat( Options::DQfD_LAMBDA_DECAY ) ) // repurposed: multiplicative
    , _margin( Options::get()->getFloat( Options::DQfD_MARGIN ) )
{
    _qNetworkLocal.to( device ).to( torch::kFloat32 );
    _qNetworkTarget.to( device ).to( torch::kFloat32 );

    // hard-sync target ← local at init (not EMA)
    {
        torch::NoGradGuard ng;
        const auto lp = _qNetworkLocal.getParameters();
        const auto tp = _qNetworkTarget.getParameters();
        for ( size_t i = 0; i < lp.size(); ++i )
            tp[i].data().copy_( lp[i].data() );
    }

    // stronger, longer DQfD: multiplicative anneal to a higher floor over ~1000 learns
    const float lambdaFloor = std::max( 0.2f, std::min( 0.5f * _lambdaSup, 0.4f ) );
    _lambdaDecay = perStepDecayToReach( _lambdaSup, lambdaFloor, /*steps*/ 1000 );

    if ( !trainedAgentPath.empty() )
        loadNetworks();
}

// ───────────────── save/load ─────────────────
void Agent::saveNetworks( const std::string &path ) const
{
    { torch::serialize::OutputArchive a; _qNetworkLocal.save( a );  a.save_to( path + "_local.pth"  ); }
    { torch::serialize::OutputArchive a; _qNetworkTarget.save( a ); a.save_to( path + "_target.pth" ); }
    DQN_LOG( "saved agent's networks" )
}

void Agent::loadNetworks()
{
    try
    {
        { torch::serialize::InputArchive a; a.load_from( _trainedAgentFilePath + "_local.pth" );  _qNetworkLocal.load( a ); }
        { torch::serialize::InputArchive a; a.load_from( _trainedAgentFilePath + "_target.pth" ); _qNetworkTarget.load( a ); }
        DQN_LOG( "loaded trained agent networks" )
    }
    catch ( const torch::Error &e )
    {
        std::cerr << "Failed to load networks: " << e.what() << std::endl;
    }
}

// ─────────────── stability utilities ───────────────
bool Agent::handleInvalidGradients()
{
    bool invalid = false;
    for ( auto &group : _optimizer.param_groups() )
        for ( auto &p : group.params() )
            if ( p.grad().defined() && ( torch::isnan( p.grad() ).any().item<bool>() ||
                                         torch::isinf( p.grad() ).any().item<bool>() ) )
            {
                std::cerr << "Invalid gradient detected, resetting gradient..." << std::endl;
                p.grad().detach_(); p.grad().zero_(); invalid = true;
            }
    return invalid;
}

// ─────────────── environment hooks ───────────────
void Agent::handleDone( const State &currentState, const unsigned numSplits )
{
    _replayedBuffer.handleDone( currentState, numSplits );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn(); // learn at episode boundary
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
                           bool isDemo )
{
    _replayedBuffer.pushActionEntry( action, previousState, numSplits, isDemo, done );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( _tStep == 0 && GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

// ─────────────── masking of illegal actions ───────────────
// Masks illegal (fixed rows or NOT_FIXED phase) with a large finite negative.
// Returns [B] bool tensor: terminal rows with no legal action.
torch::Tensor Agent::maskQInPlace( const torch::Tensor &state, torch::Tensor &Q ) const
{
    constexpr float NEG = -1e9f;
    const auto opts = Q.options();     // device + dtype
    auto s = state.to( opts );

    torch::Tensor notFixed;
    int64_t B = 1, C = (int64_t)_numPlConstraints, P = (int64_t)_numPhases;
    if ( s.dim() == 2 )
        notFixed = s.select( 1, (int64_t)DQN_RELU_NOT_FIXED_VALUE ).unsqueeze( 0 ); // [1, C]
    else
    {
        notFixed = s.select( 2, (int64_t)DQN_RELU_NOT_FIXED_VALUE ); // [B, C]
        B = notFixed.size( 0 ); C = notFixed.size( 1 );
    }

    auto rowLegal       = notFixed.gt( 0.5 );              // [B, C]
    auto terminalByMask = rowLegal.sum( 1 ).eq( 0 );       // [B]
    auto phaseIdx       = torch::arange( P, torch::TensorOptions().device( Q.device() ).dtype( torch::kLong ) ).view( { 1,1,-1 } );
    auto phaseLegal     = phaseIdx.ne( (int64_t)DQN_RELU_NOT_FIXED ); // [1,1,P]
    auto legal3D        = rowLegal.unsqueeze( 2 ) & phaseLegal;       // [B,C,P]

    if ( Q.dim() == 3 ) Q.masked_fill_( ~legal3D, NEG );                         // [B,C,P]
    else                Q.masked_fill_( ( ~legal3D ).reshape( { B, C*P } ), NEG ); // [B,A]

    return terminalByMask;
}

// ─────────────── policies ───────────────
std::unique_ptr<Action> Agent::actBestAction( const State &state )
{
    torch::NoGradGuard ng;
    _qNetworkLocal.eval();

    const auto tensorState = state.toTensor().to( device );
    auto QValues = _qNetworkLocal.forward( tensorState );
    sanitizeInPlace( QValues );
    if ( maskQInPlace( tensorState, QValues ).item<bool>() )
    {
        _qNetworkLocal.train();
        return nullptr;
    }

    unsigned actionIndex = QValues.argmax( 1 ).item<int>();
    _qNetworkLocal.train();

    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}

std::unique_ptr<Action> Agent::actRandomly( const State &state )
{
    const auto s = state.toTensor(); // [C, F] CPU
    const auto notFixedCol =
        s.index( { torch::indexing::Slice(), (int64_t)DQN_RELU_NOT_FIXED_VALUE } ).to( torch::kFloat32 );

    std::vector<unsigned> candidates; candidates.reserve( _numPlConstraints );
    for ( unsigned c = 0; c < _numPlConstraints; ++c )
        if ( notFixedCol[c].item<float>() > 0.5f ) candidates.push_back( c );

    if ( candidates.empty() ) return nullptr;

    const int pick = RandomGlobals::instance().randInt( 0, (int)candidates.size() - 1 );
    const unsigned constraint = candidates[(size_t)pick];
    const unsigned phase = ( RandomGlobals::instance().randInt( 0, 1 ) == 0 )
                             ? (unsigned)DQN_RELU_ACTIVE
                             : (unsigned)DQN_RELU_INACTIVE;

    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}

// ─────────────── learning ───────────────
void Agent::learn()
{
    auto batch = _replayedBuffer.sample();
    if ( batch.indices.empty() ) return;

    // indices [B]
    std::vector<int64_t> idx64( batch.indices.begin(), batch.indices.end() );
    auto idxTensor = torch::tensor( idx64, torch::dtype( torch::kLong ) );

    // tensors
    auto statesTensor     = _replayedBuffer.getStates().index( { idxTensor } ).to( device );
    auto actionsTensorB   = _replayedBuffer.getActions().index( { idxTensor } ).to( device ).to( torch::kLong );     // [B]
    auto rewardsTensor    = _replayedBuffer.getRewards().index( { idxTensor } ).to( device ).to( torch::kFloat32 );  // [B]
    auto nextStatesTensor = _replayedBuffer.getNextStates().index( { idxTensor } ).to( device );
    auto doneTensor       = _replayedBuffer.getDones().index( { idxTensor } ).to( device ).to( torch::kBool ).view( { -1 } );

    // Q(s,a)
    auto Qs = _qNetworkLocal.forward( statesTensor );   // [B, A]
    sanitizeInPlace( Qs );
    auto QExpected = Qs.gather( 1, actionsTensorB.view( { -1, 1 } ) ).squeeze( 1 ).to( torch::kFloat32 ); // [B]

    // ----- Double DQN targets (scoped no-grad) -----
    torch::Tensor QTargets;
    {
        torch::NoGradGuard ng;
        auto next_local = _qNetworkLocal.forward( nextStatesTensor ); // [B, A]
        sanitizeInPlace( next_local );
        auto termMask     = maskQInPlace( nextStatesTensor, next_local );
        auto next_actions = next_local.argmax( 1 ).view( { -1, 1 } ); // [B,1]

        auto next_target = _qNetworkTarget.forward( nextStatesTensor ); // [B, A]
        sanitizeInPlace( next_target );
        maskQInPlace( nextStatesTensor, next_target );
        auto targetQValuesNextState = next_target.gather( 1, next_actions ).squeeze( 1 ); // [B]

        auto notDone = ( ( ~doneTensor ) & ( ~termMask ) ).to( torch::kFloat32 );
        QTargets = rewardsTensor + GAMMA * targetQValuesNextState * notDone; // [B]
    }

    if ( torch::isnan( QTargets ).any().item<bool>() )
        throw std::runtime_error( "NaN detected in QTargets." );

    // TD loss (PER-weighted, MSE)
    auto td_errors = torch::mse_loss( QExpected, QTargets.detach(), torch::Reduction::None ); // [B]
    auto weights   = torch::tensor( batch.weights, statesTensor.options().dtype( torch::kFloat32 ) ).to( device );
    weights = weights.clamp_min( 1e-8 ); // keep scale / avoid zeros
    auto weightedTdLoss = ( td_errors * weights ).mean();

    // DQfD margin — only on demo samples
    std::vector<int64_t> demo_mask_int( batch.isDemo.begin(), batch.isDemo.end() );
    auto demo_mask = torch::tensor( demo_mask_int, torch::kLong ).to( device ).to( torch::kFloat32 ); // [B]

    torch::Tensor margin_loss = torch::zeros( {}, statesTensor.options().dtype( torch::kFloat32 ) );
    if ( demo_mask.sum().item<float>() > 0.0f )
    {
        auto all_q = _qNetworkLocal.forward( statesTensor ); // [B, A]
        sanitizeInPlace( all_q );
        maskQInPlace( statesTensor, all_q );

        auto a = actionsTensorB.view( { -1 } ); // [B]
        auto one_hot = torch::nn::functional::one_hot( a, (int64_t)_numActions )
                           .to( device ).to( torch::kFloat32 ); // [B, A]
        auto non_selected = 1.0f - one_hot;
        auto shifted      = all_q + non_selected * _margin;        // margin on non-selected
        auto max_other    = std::get<0>( shifted.max( 1 ) );        // [B]
        auto q_demo       = all_q.gather( 1, a.unsqueeze( 1 ) ).squeeze( 1 ); // [B]
        auto raw_margin   = torch::relu( max_other - q_demo ) * demo_mask;    // [B]
        margin_loss       = raw_margin.sum() / demo_mask.sum().clamp_min( 1.0f );
    }

    auto loss = weightedTdLoss + _lambdaSup * margin_loss;

    // log occasionally
    _lossVerbosity = ( _lossVerbosity + 1 ) % 100;
    if ( _lossVerbosity == 0 )
    {
        DQN_LOG( Stringf( "TD Loss: %.6f  Margin: %.6f  Total: %.6f  lambdaSup: %.3f",
                          weightedTdLoss.item<double>(),
                          margin_loss.item<double>(),
                          loss.item<double>(),
                          _lambdaSup )
                     .ascii() );
    }

    // backprop
    _optimizer.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 1.0 );
    if ( !handleInvalidGradients() )
        _optimizer.step();

    // soft target update (EMA)
    {
        torch::NoGradGuard ng2;
        softUpdate( _qNetworkLocal, _qNetworkTarget );
    }

    // update PER priorities: |TD error| + epsilon
    auto abs_td = torch::abs( QTargets.detach() - QExpected.detach() ).to( torch::kCPU ); // [B]
    auto acc = abs_td.accessor<float, 1>();
    for ( size_t i = 0; i < batch.indices.size(); ++i )
    {
        float base_eps = batch.isDemo[i] ? (float)_replayedBuffer.getEpsilonDemo()
                                         : (float)_replayedBuffer.getEpsilonAgent();
        float p = acc[i] + base_eps;
        if ( !std::isfinite( p ) ) p = base_eps;
        _replayedBuffer.updatePriority( batch.indices[i], p );
    }

    // anneal λ_sup multiplicatively during RL phase
    if ( GlobalConfiguration::DON_TRAINING_PHASE == 2 )
        _lambdaSup = std::max( 0.2f, _lambdaSup * _lambdaDecay );
}

// EMA target update with tau
void Agent::softUpdate( const QNetwork &localModel, const QNetwork &targetModel )
{
    const auto lp = localModel.getParameters();
    const auto tp = targetModel.getParameters();
    for ( size_t i = 0; i < lp.size(); ++i )
    {
        tp[i].data().copy_( GlobalConfiguration::DQN_TAU * lp[i].data() +
                            ( 1.0 - GlobalConfiguration::DQN_TAU ) * tp[i].data() );
    }
}

// ─────────────── misc ───────────────
torch::Device Agent::getDevice() const { return device; }
int Agent::getActionStackSize() const { return _replayedBuffer.getActionStackSize(); }
int Agent::getReplayBufferSize() const { return _replayedBuffer.getNumRevisitExperiences(); }
void Agent::schedulersStep() { _scheduler.step(); }
