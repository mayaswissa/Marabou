#include "DQNAgent.h"

#include <random>
#include <utility>

Agent::Agent( unsigned numPlConstraints,
              unsigned numPhases,
              const std::string &saveAgentPath,
              const std::string &trainedAgentPath )
    : _actionSpace( ActionSpace( numPlConstraints, numPhases ) )
    , _numPlConstraints( numPlConstraints )
    , _numPhaseStatuses( numPhases )
    , _embeddingDim( 4 ) // todo change
    , _numActions( _actionSpace.getSpaceSize() )
    , _qNetworkLocal( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions )
    , _qNetworkTarget( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions )
    // todo adaptive learning rate
    , optimizer( _qNetworkLocal.parameters(), torch::optim::AdamOptions( LR ).weight_decay( 1e-4 ) )
    , _replayedBuffer(
          ReplayBuffer( _numPlConstraints * _numPhaseStatuses, _numPlConstraints, BATCH_SIZE ) )
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _saveAgentFilePath( saveAgentPath )
    , _trainedAgentFilePath( trainedAgentPath )
{
    _qNetworkLocal.to( device );
    _qNetworkTarget.to( device );
    _qNetworkTarget.to( torch::kDouble );
    _qNetworkTarget.to( torch::kDouble );
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

    return Action( _numPhaseStatuses, plConstraintActionIndex, assignmentIndex );
}


void Agent::handleDone( State *currentState, bool success, unsigned stackDepth,  unsigned numSplits )
{
    // needs to insert all delayed experiences to the replay buffer and learn:
    // if done with success - the rewards of all steps in this branch remain the same.
    unsigned numExperiences = _replayedBuffer.getNumExperiences();
    if ( numExperiences > 0 )
    {
        _replayedBuffer.getExperienceAt( numExperiences - 1 )._done = true;
        _replayedBuffer.getExperienceAt( numExperiences - 1 )._reward = success ? 10 : -10;
        _replayedBuffer.moveToRevisitExperiences();
    }
    moveExperiencesToRevisitBuffer(numSplits, stackDepth, currentState);
    while ( _replayedBuffer.getNumExperiences() )
        _replayedBuffer.moveToRevisitExperiences();

    _replayedBuffer.setBufferDepth( 0 );
    learn();
}

void Agent::moveExperiencesToRevisitBuffer( unsigned currentNumSplits,
                                              unsigned depth,
                                              State *currentState )
{

    printf("insert to revisit\n");
    fflush( stdout );
    while ( _replayedBuffer.getNumExperiences() )
    {
        Experience &revisitExperience =
            _replayedBuffer.getExperienceAt( _replayedBuffer.getNumExperiences() - 1 );

        double newReward = 1.0;
        if (revisitExperience._depth == depth)
            if ( revisitExperience._currentState.getData() == currentState->getData() )
            {
                printf( "same !! \n" );
                fflush( stdout );
            }

        if ( _replayedBuffer.getExperienceBufferDepth() < depth )
            break;

        auto progress = currentNumSplits - revisitExperience._numSplits;
        if ( progress > 0 )
            newReward = 1.0 / static_cast<double>( progress );
        // newReward = static_cast<double>(_numPlConstraints - progress) / static_cast<double>(_numPlConstraints);
        revisitExperience.updateReward( newReward );
        _replayedBuffer.moveToRevisitExperiences();
    }
}

void Agent::step( State previousState,
                  Action action,
                  double reward,
                  State currentState,
                  const bool done,
                  unsigned depth,
                  unsigned numSplits,
                  bool changeReward )
{
    if ( !changeReward )
    {
        _replayedBuffer.addToRevisitExperiences(  previousState ,
                                                 action ,
                                                 static_cast<float>( reward ),
                                                 currentState ,
                                                 done,
                                                 depth,
                                                 numSplits,
                                                 changeReward );
        return;
    }

    if ( done )
    {
        _replayedBuffer.add( previousState,
                             action,
                             static_cast<float>( reward ),
                             currentState,
                             done,
                             depth,
                             numSplits,
                             changeReward );
        return;
    }

    if ( _replayedBuffer.getNumExperiences() > 0 &&
         _replayedBuffer.getExperienceBufferDepth() >= depth )
        moveExperiencesToRevisitBuffer( numSplits, depth, &currentState );

    _replayedBuffer.add( previousState,
                         action,
                         static_cast<float>( reward ),
                         currentState,
                         done,
                         depth,
                         numSplits,
                         changeReward );

    _tStep = ( _tStep + 1 ) % UPDATE_EVERY;
    if ( _tStep == 0 && _replayedBuffer.getNumRevisitExperiences() > BATCH_SIZE )
    {
        learn(); // todo Gamma should change?
    }

}


Action Agent::act( const torch::Tensor &state, double eps )
{
    _qNetworkLocal.eval();
    torch::Tensor Qvalues = _qNetworkLocal.forward( state );
    _qNetworkLocal.train();
    unsigned actionIndex;
    if ( static_cast<double>( rand() ) / RAND_MAX > eps )
        // best action - maximum Q-value :
        actionIndex = Qvalues.argmax( 1 ).item<int>();
    else
        // random :
        actionIndex = rand() % _numActions;

    auto actionIndices = _actionSpace.decodeActionIndex( actionIndex );
    return Action( _numPhaseStatuses, actionIndices.first, actionIndices.second );
}


void Agent::learn()
{
    Vector<unsigned> indices = _replayedBuffer.sample();
    if ( indices.size() < _replayedBuffer.getBatchSize() )
        return;
    std::vector<torch::Tensor> previousStates, actions, nextStates;
    std::vector<double> rewards;
    std::vector<uint8_t> dones;

    for ( const unsigned index : indices )
    {
        if ( index < _replayedBuffer.getNumRevisitExperiences() )
        {
            Experience &experience = _replayedBuffer.getRevisitExperienceAt( index );
            previousStates.push_back( experience._previousState.toTensor().to( device ) );
            actions.push_back( experience._action.actionToTensor().to( device ) );
            rewards.push_back( experience._reward );
            nextStates.push_back( experience._currentState.toTensor().to( device ) );
            dones.push_back( static_cast<uint8_t>( experience._done ) );
        }
    }

    // Create tensors from vectors
    const auto statesTensor = torch::cat( previousStates, 0 ).to( device );
    const auto actionsTensor = torch::cat( actions, 0 ).to( device );
    const auto rewardsTensor =
        torch::tensor( rewards, torch::dtype( torch::kFloat64 ) ).to( device );
    const auto nextStatesTensor = torch::cat( nextStates, 0 );
    const auto doneTensor = torch::tensor( dones, torch::dtype( torch::kUInt8 ) ).to( device );

    // DDQN : Use local network to select the best action for next states
    const auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
    const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );

    // Use target network to calculate the Q-value of these actions
    const auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
    const auto targetQValuesNextState =
        forwardTargetNet.detach().gather( 1, localQValuesNextState.unsqueeze( -1 ) ).squeeze( -1 );
    // Calculate Q targets for current states
    const auto QTargets =
        rewardsTensor + GAMMA * targetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat64 ) );

    const auto QExpected = _qNetworkLocal.forward( statesTensor )
                               .gather( 1, actionsTensor.unsqueeze( -1 ) )
                               .squeeze( -1 )
                               .to( torch::kDouble );


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
        printf("Skipped updating weights due to invalid gradients.\n");
        fflush( stdout );
    }
    optimizer.step();
    softUpdate( _qNetworkLocal, _qNetworkTarget );
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
