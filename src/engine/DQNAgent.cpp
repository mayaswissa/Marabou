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
                           const bool isDemo )
{
    _replayedBuffer.pushActionEntry( action, previousState, numSplits, isDemo, done );
    _tStep = ( _tStep + 1 ) % Options::get()->getInt( Options::DQN_EXPLORATION_RATE );
    if ( _tStep == 0 && GlobalConfiguration::DON_TRAINING_PHASE != 0 )
        learn();
}

torch::Tensor Agent::applyActionMask( const torch::Tensor &tensorState,
                                      const torch::Tensor &QValues ) const
{
    const int64_t P = static_cast<int64_t>( _numPhases );
    torch::Tensor notFixed;
    int64_t B = 1;
    auto C = static_cast<int64_t>( _numPlConstraints );
    if ( tensorState.dim() == 2 )
    { // [C,F]
        notFixed = tensorState.select( 1, static_cast<int64_t>( DQN_RELU_NOT_FIXED_VALUE ) )
                       .unsqueeze( 0 ); // [1,C]
    }
    else
    { // [B,C,F]
        notFixed =
            tensorState.select( 2, static_cast<int64_t>( DQN_RELU_NOT_FIXED_VALUE ) ); // [B,C]
        B = notFixed.size( 0 );
        C = notFixed.size( 1 );
    }
    const auto rowLegal = notFixed.gt( 0.5 ); // [B,C] bool
    const auto phaseIdx =
        torch::arange( P, torch::TensorOptions().device( QValues.device() ).dtype( torch::kLong ) )
            .view( { 1, 1, -1 } );
    const auto phaseLegal = phaseIdx.ne( static_cast<int64_t>( DQN_RELU_NOT_FIXED ) ); // [1,1,P]
                                                                                       // bool
    const auto legal3D = rowLegal.unsqueeze( 2 ) & phaseLegal;                         // [B,C,P]
    const auto legal2D = legal3D.reshape( { B, C * P } );                              // [B,A]

    if ( QValues.dim() == 1 )
        QValues.masked_fill_( ~legal2D.view( { -1 } ), -1e9f );

    else
        QValues.masked_fill_( ~legal2D, -1e9f );

    // true where row has zero legal actions
    const auto terminalByMask = legal2D.sum( 1 ).eq( 0 );
    return terminalByMask;
}
std::unique_ptr<Action> Agent::actBestAction( const State &state )
{
    torch::NoGradGuard ng;
    _qNetworkLocal.eval();
    const auto tensorState = state.toTensor().to( device );
    const torch::Tensor QValues = _qNetworkLocal.forward( tensorState );
    _qNetworkLocal.train();

    auto termMask = applyActionMask( tensorState, QValues );
    if ( termMask.to( torch::kCPU ).item<bool>() )
    {
        std::cerr << "Error: no valid next actions!" << std::endl;
        throw std::runtime_error( "no valid next actions." );
    }

    unsigned actionIndex = QValues.argmax( 1 ).item<int>();
    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}

std::unique_ptr<Action> Agent::actRandomly( const State &state )
{
    const auto tensorState = state.toTensor();
    auto reluNotFixedColumn = tensorState.index(
        { torch::indexing::Slice(), static_cast<int64_t>( DQN_RELU_NOT_FIXED_VALUE ) } );
    auto validRandomMask = ( reluNotFixedColumn == 1 );
    const torch::Tensor validRandomIndices = validRandomMask.nonzero();

    const auto k = validRandomIndices.size( 0 );
    if ( k == 0 )
        return nullptr;

    int row = RandomGlobals::instance().randInt( 0, static_cast<int>( k ) - 1 );
    const unsigned actionConstraint = validRandomIndices.index( { row, 0 } ).item<int>();

    // pick a random phase
    const unsigned actionPhase =
        RandomGlobals::instance().randInt( RELU_PHASE_ACTIVE, RELU_PHASE_INACTIVE );

    const unsigned actionIndex = _actionSpace.getActionIndex( actionConstraint, actionPhase );
    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}


void Agent::learn()
{
    auto batch = _replayedBuffer.sample();
    if ( batch.indices.empty() )
        return;

    auto idxTensor = torch::tensor( std::vector<long>( batch.indices.begin(), batch.indices.end() ),
                                    torch::kLong )
                         .to( device );
    auto states = _replayedBuffer.getStates();
    const auto statesTensor = _replayedBuffer.getStates().index( { idxTensor } ).to( device );
    const auto actionsTensor =
        _replayedBuffer.getActions().index( { idxTensor } ).to( device ).to( torch::kLong );
    const auto rewardsTensor =
        _replayedBuffer.getRewards().index( { idxTensor } ).to( device ).to( torch::kFloat32 );
    const auto nextStatesTensor =
        _replayedBuffer.getNextStates().index( { idxTensor } ).to( device );
    const auto doneTensor =
        _replayedBuffer.getDones().index( { idxTensor } ).to( device ).to( torch::kUInt8 );
    const auto QExpected = _qNetworkLocal.forward( statesTensor )
                               .gather( 1, actionsTensor )
                               .squeeze( -1 )
                               .to( torch::kFloat32 );
    auto QTargets = rewardsTensor;

    // Double DQN : Use local network to select the best action for next states
    auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
    auto termMask = applyActionMask( nextStatesTensor, forwardLocalNet );
    auto bad = ( ( ~doneTensor ) & termMask );
    ;
    if ( bad.any().to( torch::kCPU ).item<bool>() )
    {
        std::cerr << "Error: no valid next actions!" << std::endl;
        throw std::runtime_error( "no valid next actions." );
    }
    const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );

    // Use target network to calculate the Q-value of these actions
    auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
    termMask = applyActionMask( nextStatesTensor, forwardTargetNet );
    bad = ( ( ~doneTensor ) & termMask );
    if ( bad.any().to( torch::kCPU ).item<bool>() )
    {
        std::cerr << "Error: no valid next actions!" << std::endl;
        throw std::runtime_error( "no valid next actions." );
    }
    const auto targetQValuesNextState =
        forwardTargetNet.detach().gather( 1, localQValuesNextState.unsqueeze( -1 ) ).squeeze( -1 );
    // Calculate Q targets for current states
    QTargets =
        rewardsTensor + GAMMA * targetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat32 ) );

    if ( torch::isnan( QTargets ).any().item<bool>() )
    {
        std::cerr << "Error: QTargets contains NaN values!" << std::endl;
        throw std::runtime_error( "NaN detected in QTargets." );
    }
    if ( torch::isnan( QTargets ).any().item<bool>() )
    {
        std::cerr << "Error: QTargets contains NaN values! Dumping sample data:" << std::endl;
        std::cerr << "States: " << statesTensor << std::endl;
        std::cerr << "Actions: " << actionsTensor << std::endl;
        std::cerr << "Rewards: " << rewardsTensor << std::endl;
        std::cerr << "Next States: " << nextStatesTensor << std::endl;
        throw std::runtime_error( "NaN detected in QTargets." );
    }
    // TD loss
    auto td_errors = torch::smooth_l1_loss( QExpected, QTargets.detach(), torch::Reduction::None );
    auto weights = torch::tensor( batch.weights, statesTensor.options().dtype( torch::kFloat32 ) )
                       .to( device );
    weights = weights / weights.mean().clamp_min( 1e-8 );
    auto weightedTdLoss = ( td_errors * weights ).mean();
    // Margin loss for demonstration samples
    std::vector<int64_t> demo_mask_int( batch.isDemo.begin(), batch.isDemo.end() );
    auto all_q = _qNetworkLocal.forward( statesTensor );
    termMask = applyActionMask( statesTensor, all_q );
    bad = ( ( ~doneTensor ) & termMask );
    if ( bad.any().to( torch::kCPU ).item<bool>() )
    {
        std::cerr << "Error: no valid next actions!" << std::endl;
        throw std::runtime_error( "no valid next actions." );
    }
    auto demo_mask = torch::tensor( demo_mask_int, torch::TensorOptions().dtype( torch::kInt64 ) )
                         .to( device )
                         .to( torch::kFloat32 );
    auto flat_actions = actionsTensor.squeeze( -1 );
    auto one_hot =
        torch::nn::functional::one_hot( flat_actions, static_cast<int64_t>( _numActions ) )
            .to( device )
            .to( torch::kFloat32 );
    auto non_demo_mask = ( 1.0f - one_hot );
    auto shifted = all_q + non_demo_mask * _margin;
    auto max_other = std::get<0>( shifted.max( 1 ) );
    auto q_demo = all_q.gather( 1, flat_actions.unsqueeze( 1 ) ).squeeze( 1 );
    auto raw_margin = torch::relu( max_other - q_demo ) * demo_mask;
    auto marginLoss = raw_margin.sum() / demo_mask.sum().clamp_min( 1.0 );
    // Total loss
    const auto loss = weightedTdLoss + _lambdaSup * marginLoss;
    _lossVerbosity = ( _lossVerbosity + 1 ) % 100;
    if ( _lossVerbosity == 0 )
    {
        DQN_LOG( Stringf( "TD Loss : %.10f\n", weightedTdLoss.item<double>() ).ascii() );
        DQN_LOG( Stringf( "MARGIN Loss : %.10f\n", marginLoss.item<double>() ).ascii() );
        DQN_LOG( Stringf( "Loss : %.10f\n", loss.item<double>() ).ascii() );
    }

    // Backpropagation
    _optimizer.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 1.0 );
    if ( !handleInvalidGradients() )
        _optimizer.step();
    softUpdate( _qNetworkLocal, _qNetworkTarget );

    for ( size_t i = 0; i < batch.indices.size(); ++i )
    {
        float delta = QTargets[i].item<float>() - QExpected[i].item<float>();
        float raw_p = std::fabs( delta ) + ( batch.isDemo[i] ? _replayedBuffer.getEpsilonDemo()
                                                             : _replayedBuffer.getEpsilonAgent() );
        _replayedBuffer.updatePriority( batch.indices[i], raw_p );
    }

    if ( GlobalConfiguration::DON_TRAINING_PHASE == 2 )
        _lambdaSup = std::fmax( 0.1f, _lambdaSup - _lambdaDecay );
}


void Agent::softUpdate( const QNetwork &localModel, QNetwork &targetModel )
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