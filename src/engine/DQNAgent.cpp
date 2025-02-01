#include "DQNAgent.h"

#include <random>
#include <utility>

Agent::Agent( const unsigned numPlConstraints,
              const unsigned numPhases,
              const std::string &saveAgentPath,
              const std::string &trainedAgentPath )
    : _actionSpace( ActionSpace( numPlConstraints, numPhases ) )
    , _numPlConstraints( numPlConstraints )
    , _numPhaseStatuses( numPhases )
    , _embeddingDim( 4 ) // todo change
    , _numActions( _actionSpace.getSpaceSize() )
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _saveAgentFilePath( saveAgentPath )
    , _trainedAgentFilePath( trainedAgentPath )
    , _qNetworkLocal( QNetwork( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions ) )
    , _qNetworkTarget(
          QNetwork( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions ) )
    , optimizer( _qNetworkLocal.parameters(),
                 torch::optim::AdamOptions( GlobalConfiguration::DQN_LR ).weight_decay( 1e-4 ) )
    , _replayedBuffer( ReplayBuffer( _numPlConstraints * _numPhaseStatuses, 10000, _batchSize ) )
{
    _qNetworkLocal.to( device );
    _qNetworkTarget.to( device );
    _qNetworkLocal.to( torch::kFloat32 );
    _qNetworkTarget.to( torch::kFloat32 );
    // If a load path is provided, load the networks
    if ( !trainedAgentPath.empty() )
    {
        loadNetworks();
    }
}

void Agent::saveNetworks() const
{
    torch::serialize::OutputArchive output_archive;
    _qNetworkLocal.save( output_archive );
    output_archive.save_to( _saveAgentFilePath + "_local.pth" );
    _qNetworkTarget.save( output_archive );
    output_archive.save_to( _saveAgentFilePath + "_target.pth" );
}


void Agent::loadNetworks()
{
    try
    {
        torch::serialize::InputArchive input_archive;
        input_archive.load_from( _trainedAgentFilePath + "_local.pth" );
        _qNetworkLocal.load( input_archive );
        input_archive.load_from( _trainedAgentFilePath + "_target.pth" );
        _qNetworkTarget.load( input_archive );
    }
    catch ( const torch::Error &e )
    {
        std::cerr << "Failed to load networks: " << e.what() << std::endl;
    }
}

bool Agent::handleInvalidGradients()
{
    bool invalid = false;
    for ( auto &group : optimizer.param_groups() )
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

Action Agent::tensorToAction( const torch::Tensor &tensor ) const
{
    int combinedIndex = tensor.item<int>();

    int plConstraintActionIndex = combinedIndex / _numPhaseStatuses;
    int assignmentIndex = combinedIndex % _numPhaseStatuses;

    return Action( _numPhaseStatuses, _numPlConstraints, plConstraintActionIndex, assignmentIndex );
}


void Agent::handleDone( const State &currentState,
                        const unsigned stackDepth,
                        const unsigned numSplits,
                        const double prunedSubtrees )
{
    // Insert all actions from actions buffer to the replay buffer and learn.
    _replayedBuffer.handleDone( currentState, stackDepth, numSplits, prunedSubtrees );
    learn();
}

void Agent::addAlternativeAction( const State &stateBeforeSplit,
                                  const unsigned depthBeforeSplit,
                                  const unsigned numSplits,
                                  unsigned &numInconsistent,
                                  const double prunedSubtrees )
{
    _replayedBuffer.applyNextAction(
        stateBeforeSplit, depthBeforeSplit, numSplits, numInconsistent, prunedSubtrees );
    _tStep = ( _tStep + 1 ) % _updateEvery;
    if ( _tStep == 0 && _replayedBuffer.getNumRevisitExperiences() > _batchSize )
        learn();
}


void Agent::step( const State &previousState,
                  const Action &action,
                  const double reward,
                  const State &currentState,
                  const bool done,
                  const unsigned depth,
                  const unsigned numSplits,
                  const bool changeReward )
{
    // invalid step due to fixed pl constraint or not fixed phase in action.
    if ( !changeReward || done )
        _replayedBuffer.addToRevisitExperiences( previousState,
                                                 action,
                                                 static_cast<float>( reward ),
                                                 currentState,
                                                 done,
                                                 depth,
                                                 numSplits,
                                                 changeReward );
    else
    {
        _replayedBuffer.pushActionEntry( action, previousState, currentState, depth, numSplits );
        _tStep = ( _tStep + 1 ) % _updateEvery;
        if ( _tStep == 0 && _replayedBuffer.getNumRevisitExperiences() > _batchSize )
            learn();
    }
}

Action Agent::act( const State &state, const double eps )
{
    _qNetworkLocal.eval();
    torch::Tensor QValues = _qNetworkLocal.forward( state.toTensor() );
    _qNetworkLocal.train();
    unsigned actionIndex;

    // Create a mask to invalidate actions with phase not fixed or already fixed pl-constraint
    torch::Tensor mask = torch::zeros( { _numActions } );
    for ( unsigned i = 0; i < _numPlConstraints; i++ )
    {
        const unsigned index = i * _numPhaseStatuses;
        mask[index] = -std::numeric_limits<float>::infinity();

        if ( state.getData()[i][PHASE_NOT_FIXED] == 0 ) // plConstraint in current state is fixed -
                                                        // invalid action.
        {
            for ( unsigned j = 0; j < _numPhaseStatuses; j++ )
            {
                mask[i * _numPhaseStatuses + j] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Apply the mask
    QValues += mask;

    if ( static_cast<double>( rand() ) / RAND_MAX > eps )
    {
        {
            // best action - maximum Q-value from the masked values
            actionIndex = QValues.argmax().item<int>();
            printf( "chose by agent\n" );
            fflush( stdout );
        }
    }
    else
    {
        // printf( "chose randomly\n" );
        // fflush( stdout );
        std::vector<unsigned> validConstraints;
        for ( unsigned i = 0; i < _numPlConstraints; ++i )
        {
            if ( state.getData()[i][PHASE_NOT_FIXED] != 0 )
                validConstraints.push_back( i );
        }
        unsigned actionConstraint = validConstraints[rand() % validConstraints.size()];
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_int_distribution<> dist( RELU_PHASE_ACTIVE, RELU_PHASE_INACTIVE );
        unsigned actionPhase = dist( gen );
        actionIndex = _actionSpace.getActionIndex( actionConstraint, actionPhase );
    }

    auto actionIndices = _actionSpace.decodeActionIndex( actionIndex );
    return Action(
        _numPhaseStatuses, _numPlConstraints, actionIndices.first, actionIndices.second );
}


void Agent::learn()
{
    Vector<unsigned> indices = _replayedBuffer.sample();
    if ( indices.size() < _replayedBuffer.getBatchSize() || indices.empty() )
        return;
    std::vector<torch::Tensor> previousStates, actions, nextStates;
    std::vector<double> rewards;
    std::vector<uint8_t> dones;

    for ( const unsigned index : indices )
    {
        if ( index < _replayedBuffer.getNumRevisitExperiences() )
        {
            Experience &experience = _replayedBuffer.getRevisitExperienceAt( index );
            previousStates.push_back(
                experience._stateBeforeAction.toTensor().unsqueeze( 0 ).to( device ) );
            actions.push_back( experience._action.actionToTensor().to( device ) );
            rewards.push_back( experience._reward );
            nextStates.push_back(
                experience._stateAfterAction.toTensor().unsqueeze( 0 ).to( device ) );
            dones.push_back( static_cast<uint8_t>( experience._done ) );
        }
    }

    // Concatenate tensors along the batch dimension
    const auto statesTensor = torch::cat( previousStates, 0 );
    const auto actionsTensor = torch::cat( actions, 0 ).view( { -1, 1 } );
    const auto rewardsTensor =
        torch::tensor( rewards, torch::dtype( torch::kFloat32 ) ).to( device );
    const auto nextStatesTensor = torch::cat( nextStates, 0 );
    const auto doneTensor = torch::tensor( dones, torch::dtype( torch::kUInt8 ) ).to( device );

    auto QExpected = _qNetworkLocal.forward( statesTensor )
                         .gather( 1, actionsTensor )
                         .squeeze( -1 )
                         .to( torch::kFloat32 );
    auto QTargets = rewardsTensor;

    if ( GlobalConfiguration::DQN_TRAINING )
    {
        // Double DQN : Use local network to select the best action for next states
        const auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
        const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );

        // Use target network to calculate the Q-value of these actions
        const auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
        const auto targetQValuesNextState = forwardTargetNet.detach()
                                                .gather( 1, localQValuesNextState.unsqueeze( -1 ) )
                                                .squeeze( -1 );
        // Calculate Q targets for current states
        QTargets = rewardsTensor +
                   GAMMA * targetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat32 ) );
        std::cout << "QTargets min: " << QTargets.min().item<double>()
                  << ", max: " << QTargets.max().item<double>() << std::endl;
        QExpected = _qNetworkLocal.forward( statesTensor )
                        .gather( 1, actionsTensor )
                        .squeeze( -1 )
                        .to( torch::kFloat32 );

        if ( torch::isnan( QTargets ).any().item<bool>() )
        {
            std::cerr << "Error: QTargets contains NaN values!" << std::endl;
            throw std::runtime_error( "NaN detected in QTargets." );
        }
        for ( const auto &param : _qNetworkLocal.parameters() )
        {
            if ( param.grad().defined() && torch::isnan( param.grad() ).any().item<bool>() )
            {
                std::cerr << "Error: NaN detected in gradients!" << std::endl;
                throw std::runtime_error( "NaN gradients detected." );
            }
        }

        const auto loss = torch::mse_loss( QExpected, QTargets );
        printf( "Loss: %f\n", loss.item<double>() );

        // Backpropagation
        optimizer.zero_grad();
        loss.backward();
        if ( !handleInvalidGradients() )
        {
            optimizer.step();
        }
        else
        {
            printf( "Skipped updating weights due to invalid gradients.\n" );
            fflush( stdout );
        }
        optimizer.step();
        softUpdate( _qNetworkLocal, _qNetworkTarget );
    }
    else
    {
        const auto loss = torch::mse_loss( QExpected, QTargets );
        printf( "Validation Loss: %f\n", loss.item<double>() );
    }
}


void Agent::softUpdate( const QNetwork &localModel, const QNetwork &targetModel )
{
    const auto localParams = localModel.getParameters();
    const auto targetParams = targetModel.getParameters();
    for ( size_t i = 0; i < localParams.size(); ++i )
    {
        targetParams[i].data().copy_( TAU * localParams[i].data() +
                                      ( 1.0 - TAU ) * targetParams[i].data() );
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
