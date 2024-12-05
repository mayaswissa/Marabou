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
    , _replayedBuffer( _numPlConstraints * _numPhaseStatuses, _numPlConstraints, BATCH_SIZE )
    , _tStep( 0 )
    , _experienceDepth( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _saveAgentFilePath( saveAgentPath )
    , _trainedAgentFilePath( trainedAgentPath )
{
    _qNetworkLocal.to( device );
    _qNetworkTarget.to( device );
    _qNetworkTarget.to( torch::kDouble );
    _qNetworkTarget.to( torch::kDouble );
    // _delayedReplayBuffer = DelayedReplayBuffer();
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

bool Agent::handle_invalid_gradients()
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
Action Agent::tensorToAction( const torch::Tensor &tensor )
{
    int combinedIndex = tensor.item<int>();

    int plConstraintActionIndex = combinedIndex / _numPhaseStatuses;
    int assignmentIndex = combinedIndex % _numPhaseStatuses;

    return Action( _numPhaseStatuses, plConstraintActionIndex, assignmentIndex );
}

unsigned Agent::getNumExperiences() const
{
    return _replayedBuffer.getNumRevisitExperiences();
}


void Agent::handleDone( bool success )
{
    // todo - change last action's done to true , change its reward to a good reward if done with
    // success and bad if not. todo - add all experiences to revisited experiences. todo - change
    // Buffer size. todo - check why not equal state found.

    // needs to insert all delayed experiences to the replay buffer and learn:
    // if done with success - the rewards of all steps in this branch remain the same.
    if ( _replayedBuffer.numExperiences() )
    {
        _replayedBuffer.updateReturnedWhenDoneSuccess();
        _replayedBuffer.getExperienceAt( -1 ).done = true;
        _replayedBuffer.getExperienceAt( -1 ).reward = success ? 10 : -10;
        _replayedBuffer.moveToRevisitExperiences();
    }
    while ( _replayedBuffer.numExperiences() )
        _replayedBuffer.moveToRevisitExperiences();

    learn( GAMMA );
}

void Agent::moveExperiencesToRevisitedBuffer( unsigned currentNumSplits,
                                              unsigned depth,
                                              State * /*state*/ )
{
    printf( "backward\n" );
    fflush( stdout );

    int skippedSplits = -1;
    while ( _replayedBuffer.numExperiences() )
    {
        skippedSplits++;
        Experience &previousExperience =
            _replayedBuffer.getExperienceAt( _replayedBuffer.numExperiences() - 1 );

        if ( _experienceDepth < depth )
            break;

        double newReward = 1;
        // if (previousExperience.state.getData() == state->getData())
        // {
        _experienceDepth = previousExperience.depth;
        auto progress = currentNumSplits - previousExperience.numSplits;
        if ( progress > 0 )
            newReward = 1.0 / static_cast<double>( progress ) * 100;
        printf( "same!!, reward : %f\n", newReward );
        fflush( stdout );
        // } else
        // {
        //     printf("not the same, reward : %f\n", newReward);
        //     fflush(stdout);
        //     // todo?
        // }
        previousExperience.updateReward( newReward );


        _replayedBuffer.moveToRevisitExperiences();
    }
}

void Agent::addToExperiences( State state,
                              Action action,
                              double reward,
                              State nextState,
                              const bool done,
                              unsigned depth,
                              unsigned numSplits,
                              bool changeReward )
{
    // need to first check if the depth is highe to the last inserted depth.
    // if so - go over all the experiences until this depth's experience, give a reward and change
    // vector. then when reaching the same depth validate its the same phase and plConstraint, if so
    // - the reward would be the diff of depth / plConstraints.size (?) from the last inserted. if
    // not - dont know.
    // if not - if its equal - do something
    // if not and not - insert it to the regular exeperiences vector

    if ( done )
    {
        _replayedBuffer.addToRevisitExperiences( state,
                                                 action,
                                                 static_cast<float>( reward ),
                                                 nextState,
                                                 done,
                                                 depth,
                                                 numSplits,
                                                 changeReward );
        handleDone( done );
        return;
    }
    if ( !changeReward )
    {
        _replayedBuffer.addToRevisitExperiences( state,
                                                 action,
                                                 static_cast<float>( reward ),
                                                 nextState,
                                                 done,
                                                 depth,
                                                 numSplits,
                                                 changeReward );
        return;
    }

    if ( _replayedBuffer.numExperiences() > 0 && _experienceDepth >= depth )
        moveExperiencesToRevisitedBuffer( numSplits, depth, &state );

    _replayedBuffer.add( state,
                         action,
                         static_cast<float>( reward ),
                         nextState,
                         done,
                         depth,
                         numSplits,
                         changeReward );
    _experienceDepth = depth;
}

// void Agent::addToExperiences( unsigned /*currentNumSplits*/,
//                               State state,
//                               Action action,
//                               double reward,
//                               State nextState,
//                               const bool done,
//                               unsigned depth,
//                               unsigned numSplits,
//                               bool changeReward )
// {
//     // need to first check if the depth is highe to the last inserted depth.
//     // if so - go over all the experiences until this depth's experience, give a reward and
//     change
//     // vector. then when reaching the same depth validate its the same phase and plConstraint, if
//     so
//     // - the reward would be the diff of depth / plConstraints.size (?) from the last inserted.
//     if
//     // not - dont know.
//     // if not - if its equal - do something
//     // if not and not - insert it to the regular exeperiences vector
//
//
//     if ( !changeReward )
//     {
//         _replayedBuffer.addToRevisitExperiences( state,
//                                                  action,
//                                                  static_cast<float>( reward ),
//                                                  nextState,
//                                                  done,
//                                                  depth,
//                                                  numSplits,
//                                                  changeReward );
//         return;
//     }
//
//     if ( _replayedBuffer.numExperiences() == 0 || _experienceDepth < depth )
//     {
//         _replayedBuffer.add( state,
//                              action,
//                              static_cast<float>( reward ),
//                              nextState,
//                              done,
//                              depth,
//                              numSplits,
//                              changeReward );
//         _experienceDepth = depth;
//         return;
//     }
//
//
//     int skippedSplits = -1;
//     while ( _replayedBuffer.numExperiences() )
//     {
//         skippedSplits++;
//         Experience &previousExperience =
//             _replayedBuffer.getExperienceAt( _replayedBuffer.numExperiences() - 1 );
//         _experienceDepth = previousExperience.depth;
//         if ( _experienceDepth < depth )
//             break;
//         // todo - check if possible to skip the relu and go to a deeper depth in its other
//         // encourage the agent to quickly prune branches
//         double splitsSkipped = _numPlConstraints - _experienceDepth -
//                                skippedSplits;
//         double newReward = static_cast<double>( splitsSkipped / _numPlConstraints ); // todo?
//         printf( "new reward : %f\n", newReward );
//         fflush( stdout );
//         previousExperience.updateReward( newReward );
//
//         // auto progress = currentNumSplits - experience.numSplits;
//         // double reward = 0.0;
//         // if (progress > 0)
//         //     reward = 1.0 / static_cast<double>(progress);
//
//         // experience.updateReward( reward );
//         _replayedBuffer.moveToRevisitExperiences();
//     }
// }

void Agent::step()
{
    _tStep = ( _tStep + 1 ) % UPDATE_EVERY;
    if ( _tStep == 0 && _replayedBuffer.numRevisitedExperiences() > BATCH_SIZE )
    {
        learn( GAMMA ); // todo Gamma should change?
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


void Agent::learn( const double gamma )
{
    Vector<unsigned> indices = _replayedBuffer.sample();
    if ( indices.empty() )
        return;
    std::vector<torch::Tensor> states, actions, nextStates;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;
    // for index in indices - get the specific experience from memory .
    // then one more loop - to delete them ? no need, maybe.
    for ( const unsigned index : indices )
    {
        if ( index < _replayedBuffer.numRevisitedExperiences() )
        {
            Experience &experience = _replayedBuffer.getRevisitedExperienceAt( index );
            states.push_back( experience.state.toTensor().to( device ) );
            actions.push_back( experience.action.actionToTensor().to( device ) );
            rewards.push_back( experience.reward );
            nextStates.push_back( experience.nextState.toTensor().to( device ) );
            dones.push_back( static_cast<uint8_t>( experience.done ) );
        }
    }
    // Create tensors from vectors
    const auto statesTensor = torch::cat( states, 0 ).to( device );
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
        rewardsTensor + gamma * targetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat64 ) );

    const auto QExpected = _qNetworkLocal.forward( statesTensor )
                               .gather( 1, actionsTensor.unsqueeze( -1 ) )
                               .squeeze( -1 )
                               .to( torch::kDouble );


    const auto loss = torch::mse_loss( QExpected, QTargets );
    printf( "Loss: %f\n", loss.item<double>() );

    // Backpropagation
    optimizer.zero_grad();
    loss.backward();
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
