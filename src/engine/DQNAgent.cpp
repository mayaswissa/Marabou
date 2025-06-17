#include "DQNAgent.h"

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
    , _qNetworkLocal( QNetwork( _numPlConstraints, NUM_FEATURES, _numActions ) )
    , _qNetworkTarget( QNetwork( _numPlConstraints, NUM_FEATURES, _numActions ) )
    , _optimizer( _qNetworkLocal.parameters(),
                  torch::optim::AdamOptions( GlobalConfiguration::DQN_LR ).weight_decay( 1e-5 ) )
    , _scheduler( _optimizer, 1, 0.9 )
    , _replayedBuffer( ReplayBuffer( _numPlConstraints,
                                     GlobalConfiguration::DQN_BUFFER_SIZE,
                                     GlobalConfiguration::DQN_BATCH_SIZE ) )
    , _lossVerbosity( 0 )
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
    _tStep = ( _tStep + 1 ) % GlobalConfiguration::DQN_EXPLORATION_RATE;
    learn();
}

void Agent::stepAlternativeAction( const State &stateBeforeSplit,
                                   const unsigned numSplits,
                                   unsigned &numInconsistent )
{
    _replayedBuffer.applyNextAction( stateBeforeSplit, numSplits, numInconsistent );
    _tStep = ( _tStep + 1 ) % GlobalConfiguration::DQN_EXPLORATION_RATE;
    if ( _tStep == 0 )
        learn();
}

void Agent::stepFakeAction( const State &stateBeforeAction, const unsigned numSplitsBeforeAction )
{
    _replayedBuffer.pushFakeActionEntry( stateBeforeAction, numSplitsBeforeAction );
}

void Agent::stepNewAction( const State &previousState,
                           const Action &action,
                           const bool done,
                           const unsigned numSplits )
{
    _replayedBuffer.pushActionEntry( action, previousState, numSplits, done );
    _tStep = ( _tStep + 1 ) % GlobalConfiguration::DQN_EXPLORATION_RATE;
    if ( _tStep == 0 )
        learn();
}

std::unique_ptr<Action> Agent::act( const State &state, const double eps )
{
    _qNetworkLocal.eval();
    const auto tensorState = state.toTensor();
    torch::Tensor QValues = _qNetworkLocal.forward( tensorState );
    _qNetworkLocal.train();
    unsigned actionIndex;

    // Create a mask to invalidate actions with phase not fixed or already fixed pl-constraint
    auto mask = torch::zeros( { static_cast<long>( _numActions ) }, torch::kFloat32 );
    auto mask2D =
        mask.view( { static_cast<long>( _numPlConstraints ), static_cast<long>( _numPhases ) } );

    // can not choose to convert a constraint back to an unfixed phase.
    mask2D.index_put_( { torch::indexing::Slice(), static_cast<int64_t>( DQN_RELU_NOT_FIXED ) },
                       -std::numeric_limits<float>::infinity() );

    // can not choose to change a fixed constraint.
    auto reluNotFixedColumn = tensorState.index(
        { torch::indexing::Slice(), static_cast<int64_t>( DQN_RELU_NOT_FIXED ) } );
    auto fixedMask = ( reluNotFixedColumn == 0 );
    auto expandedMask = fixedMask.unsqueeze( 1 ).expand( { -1, static_cast<long>( _numPhases ) } );

    mask2D.masked_fill_( expandedMask, -std::numeric_limits<float>::infinity() );

    auto maskFlat = mask2D.view( { -1 } );
    QValues += maskFlat;

    if ( static_cast<double>( rand() ) / RAND_MAX > eps )
    {
        // best action - maximum Q-value from the masked values
        actionIndex = QValues.argmax().item<int>();
    }
    else
    {
        auto validRandomMask = ( reluNotFixedColumn == 1 );
        torch::Tensor validRandomIndices = validRandomMask.nonzero();
        if ( validRandomIndices.size( 0 ) == 0 )
            return nullptr;
        std::random_device rd;
        std::mt19937 gen( rd() );
        int k = validRandomIndices.size( 0 ); // number of “not fixed” constraints
        std::uniform_int_distribution<> pickDist( 0, k - 1 );
        int row = pickDist( gen );
        const unsigned actionConstraint = validRandomIndices.index( { row, 0 } ).item<int>();
        std::uniform_int_distribution<> dist( RELU_PHASE_ACTIVE, RELU_PHASE_INACTIVE );
        const unsigned actionPhase = dist( gen );
        actionIndex = _actionSpace.getActionIndex( actionConstraint, actionPhase );
    }

    auto [constraint, phase] = _actionSpace.decodeActionIndex( actionIndex );
    return std::make_unique<Action>( _numPhases, _numPlConstraints, constraint, phase );
}


void Agent::learn()
{
    std::vector<unsigned> indices = _replayedBuffer.sample();
    if ( indices.empty() )
        return;
    const std::vector<long> idxLong( indices.begin(), indices.end() );

    auto idxTensor = torch::tensor( idxLong, torch::kLong ).to( device );
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
    const auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
    const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );

    // Use target network to calculate the Q-value of these actions
    const auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
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

    const auto loss = torch::mse_loss( QExpected, QTargets );
    _lossVerbosity = ( _lossVerbosity + 1 ) % 100;
    if ( _lossVerbosity == 0 )
        DQN_LOG( Stringf( "MSE Loss : %f\n", loss.item<double>() ).ascii() );

    // Backpropagation
    _optimizer.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 1.0 );
    if ( !handleInvalidGradients() )
        _optimizer.step();
    softUpdate( _qNetworkLocal, _qNetworkTarget );
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
