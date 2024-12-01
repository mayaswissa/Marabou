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
    , _experiences( _numPlConstraints * _numPhaseStatuses, 1e5, BATCH_SIZE )
    , _delayedExperiences()
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
    , _filePath( trainedAgentPath )
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
    _delayedExperiences.addExperience( state, action, reward, nextState, done, depth, numSplits );
    if ( done )
    {
        // needs to insert all delayed experiences to the replay buffer and learn.
        for ( auto delayedExperience : _delayedExperiences )
            _experiences.add( delayedExperience.getExperience() );


        const auto experiences = _experiences.sample();
        learn( GAMMA );
    }
}
void Agent::step( int currentDepth, unsigned numSplits )
{
    // go over all steps with depth >=  currentDepth and move to replay memory with reward =  1 /
    // delay in splits
    while ( _delayedExperiences.getSize() > 0 && _delayedExperiences.getDepth() <= currentDepth )
    {
        DelayedExperience delayedExperience = _delayedExperiences.popLast();
        // todo - check if possible to skip the relu and go to a deeper depth in its other
        // assignment if no progress in splits - action was invalid - reward stays the same
        if ( numSplits > delayedExperience._delay ) // todo change to get delay fucntion
        {
            double const reward = static_cast<double>( numSplits - delayedExperience._delay );
            delayedExperience._experience.updateReward( reward );
        }
        Experience experience = delayedExperience.getExperience();
        _experiences.add( experience );
    }
    _tStep = ( _tStep + 1 ) % UPDATE_EVERY; // todo check
    if ( _tStep == 0 && _experiences.size() > BATCH_SIZE )
    {
        const auto experiences = _experiences.sample();
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


void Agent::learn(  const double gamma )
{
    Vector<unsigned> indices  = _experiences.sample();
    std::vector<torch::Tensor> states, actions, nextStates;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;
    // for index in indices - get the specific experience from memory .
    // then one more loop - to delete them ? no need, maybe.
    for ( unsigned index : indices )
    {
        if (index <  _experiences.size())
        {
            Experience experience = _experiences.getExperienceAt( index);
            states.push_back( experience.state.to( device ) );
            actions.push_back( experience.action.to( device ) );
            rewards.push_back( experience.reward );
            nextStates.push_back( experience.nextState.to( device ) );
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

    auto mean = rewardsTensor.mean();
    auto std = rewardsTensor.std().clamp_min( 1e-5 );
    // Normalize rewards
    auto normalized_rewards = ( rewardsTensor - mean ) / std;

    // DDQN : Use local network to select the best action for next states
    const auto forwardLocalNet = _qNetworkLocal.forward( nextStatesTensor );
    const auto localQValuesNextState = forwardLocalNet.detach().argmax( 1 );

    // Use target network to calculate the Q-value of these actions
    const auto forwardTargetNet = _qNetworkTarget.forward( nextStatesTensor );
    const auto targetQValuesNextState =
        forwardTargetNet.detach().gather( 1, localQValuesNextState.unsqueeze( -1 ) ).squeeze( -1 );
    // Calculate Q targets for current states
    const auto QTargets = normalized_rewards +
                          gamma * targetQValuesNextState * ( 1 - doneTensor.to( torch::kFloat64 ) );

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
