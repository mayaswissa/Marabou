#include "DQNAgent.h"

#include <random>
#include <utility>

Agent::Agent( const ActionSpace &actionSpace, const std::string &trainedAgentPath )
    : _actionSpace( actionSpace )
    , _numPlConstraints( actionSpace.getNumConstraints() )
    , _numPhaseStatuses( actionSpace.getNumPhases() )
    , _embeddingDim( 4 ) // todo change
    , _numActions( actionSpace.getSpaceSize() )
    , _qNetworkLocal( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions )
    , _qNetworkTarget( _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions )
    // todo adaptive learning rate
    , optimizer( _qNetworkLocal.parameters(), torch::optim::AdamOptions( LR ).weight_decay( 1e-4 ) )
    , _memory( _numPlConstraints * _numPhaseStatuses, 1e5, BATCH_SIZE )
    , _delayedReplayBuffer()
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _filePath( trainedAgentPath )
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

void Agent::saveNetworks( const std::string &filepath ) const
{
    printf( "Saving networks...\n" );
    fflush( stdout );
    torch::serialize::OutputArchive output_archive;
    _qNetworkLocal.save( output_archive );
    output_archive.save_to( filepath + "_local.pth" );
    _qNetworkTarget.save( output_archive );
    output_archive.save_to( filepath + "_target.pth" );
}

void Agent::loadNetworks()
{
    try
    {
        printf( "Loading networks..." );
        fflush( stdout );
        torch::serialize::InputArchive input_archive;
        input_archive.load_from( _filePath + "_local.pth" );
        _qNetworkLocal.load( input_archive );
        input_archive.load_from( _filePath + "_target.pth" );
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

void Agent::AddToDelayBuffer( const torch::Tensor &state,
                  const torch::Tensor &action,
                  const double reward,
                  const torch::Tensor &nextState,
                  const bool done,
                  unsigned depth,
                  unsigned numSplits )
{
    // save the experience in the delayed replay memory :
    _delayedReplayBuffer.addExperience( state, action, reward, nextState, done, depth, numSplits );
}
void Agent::step( unsigned currentDepth, unsigned numSplits )
{
    // go over all steps with depth >=  currentDepth and move to replay memory with reward =  1 / delay in splits
    while (_delayedReplayBuffer.getSize() >0 &&_delayedReplayBuffer.getDepth() <= currentDepth)
    {
        DelayedExperience delayedExperience = _delayedReplayBuffer.popLast();
        // todo - check if possible to skip the root and go to a higher depth in the second branch
        double reward = static_cast<double> (numSplits - delayedExperience._delay);
        delayedExperience._experience->updateReward( reward );
        _memory.add( delayedExperience.getExperience() );
    }
    _tStep = ( _tStep + 1 ) % UPDATE_EVERY; // todo check
    if ( _tStep == 0 && _memory.size() > BATCH_SIZE )
    {
        const auto experiences = _memory.sample();
        learn( experiences, GAMMA );
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


void Agent::learn( Vector<std::unique_ptr<Experience>> experiences, const double gamma )
{
    std::vector<torch::Tensor> states, actions, nextStates;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;
    for ( const std::unique_ptr<Experience> experience : experiences )
    {
        states.push_back( experience->state.to( device ) );
        actions.push_back( experience->action.to( device ) );
        rewards.push_back( experience->reward );
        nextStates.push_back( experience->nextState.to( device ) );
        dones.push_back( static_cast<uint8_t>( experience->done ) );
    }

    // Create tensors from vectors
    const auto statesTensor = torch::cat( states, 0 ).to( device );
    const auto actionsTensor = torch::cat( actions, 0 ).to( device );
    const auto rewardsTensor =
        torch::tensor( rewards, torch::dtype( torch::kFloat64 ) ).to( device );
    // Calculate mean and std of rewards
    auto mean = rewardsTensor.mean();
    auto std = rewardsTensor.std().clamp_min( 1e-5 );
    ;
    // Normalize rewards
    auto normalized_rewards = ( rewardsTensor - mean ) / std;

    const auto nextStatesTensor = torch::cat( nextStates, 0 );
    const auto doneTensor = torch::tensor( dones, torch::dtype( torch::kUInt8 ) ).to( device );
    for ( const auto &param : _qNetworkLocal.parameters() )
    {
        if ( torch::isnan( param ).any().item<bool>() || torch::isinf( param ).any().item<bool>() )
        {
            printf( "NaN detected in local network parameters. \n" );
            fflush( stdout );
        }
    }
    for ( const auto &param : _qNetworkTarget.parameters() )
    {
        if ( torch::isnan( param ).any().item<bool>() || torch::isinf( param ).any().item<bool>() )
        {
            printf( "NaN detected in target network parameters. \n" );
            fflush( stdout );
        }
    }

    // DDQN : Use local network to select the best action for next states
    const auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
    if ( torch::isnan( forwardLocalNet ).any().item<bool>() )
    {
        printf( "NaN detected in Q-values from the local network.\n" );
        fflush( stdout );
    }
    const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );


    // Use target network to calculate the Q-value of these actions
    const auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
    if ( torch::isnan( forwardTargetNet ).any().item<bool>() )
    {
        printf( "NaN detected in Q-values from the target network.\n" );
        fflush( stdout );
    }
    const auto taegetQValuesNextState =
        forwardTargetNet.detach().gather( 1, localQValuesNextState.unsqueeze( -1 ) ).squeeze( -1 );

    // Calculate Q targets for current states
    const auto QTargets = normalized_rewards +
                          gamma * taegetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat64 ) );

    const auto QExpected = _qNetworkLocal.forward( statesTensor )
                               .gather( 1, actionsTensor.unsqueeze( -1 ) )
                               .squeeze( -1 )
                               .to( torch::kDouble );


    const auto loss = torch::mse_loss( QExpected, QTargets );
    printf( "Loss: %f\n", loss.item<double>() );
    if ( torch::isnan( loss ).any().item<bool>() || torch::isinf( loss ).any().item<bool>() )
    {
        printf( "Detected NaN or Inf in loss\n" );
        fflush( stdout );
    }

    // Backpropagation
    optimizer.zero_grad();
    loss.backward();
    // torch::nn::utils::clip_grad_norm_( _qNetworkLocal.parameters(), 5.0 );
    // if ( !handle_invalid_gradients() )
    // {
    //     optimizer.step();
    // }
    // else
    // {
    //     printf("Skipped updating weights due to invalid gradients.\n");
    //     fflush( stdout );
    // }

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
