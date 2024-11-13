#include "DQNAgent.h"

#include <random>
#include <utility>

Agent::Agent( const ActionSpace &actionSpace )
    : _actionSpace( actionSpace )
    , _numVariables( actionSpace.getNumConstraints() )
    , _numPhaseStatuses( actionSpace.getNumPhases() )
    , _embeddingDim( 4 )
    , _numActions( actionSpace.getSpaceSize() )
    , _qNetworkLocal( _numVariables, _numPhaseStatuses, _embeddingDim, _numActions )
    , _qNetworkTarget( _numVariables, _numPhaseStatuses, _embeddingDim, _numActions )
    , optimizer( _qNetworkLocal.parameters(), torch::optim::AdamOptions( LR ) )
    , _memory( _numVariables * _numPhaseStatuses, 1e5, BATCH_SIZE )
    , _tStep( 0 )
    , device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU )
{
    _qNetworkLocal.to( device );
    _qNetworkTarget.to( device );
}


Action Agent::tensorToAction( const torch::Tensor &tensor )
{
    const unsigned int constraintIndex = tensor[0].item<int>();
    const auto assignment = static_cast<PhaseStatus>( tensor[1].item<int>() );
    return Action( constraintIndex, assignment );
}

void Agent::step( const torch::Tensor &state,
                  const torch::Tensor &action,
                  const unsigned reward,
                  const torch::Tensor &nextState,
                  const bool done )
{
    // save experience in replay memory
    _memory.add( state, action, reward, nextState, done );
    _tStep = ( _tStep + 1 ) % UPDATE_EVERY;
    if ( _tStep == 0 && _memory.size() > BATCH_SIZE )
    {
        const auto experiences = _memory.sample();
        learn( experiences, GAMMA );
    }
}
Action Agent::act( const torch::Tensor &state, float eps )
{
    _qNetworkLocal.eval();
    torch::Tensor Qvalues = _qNetworkLocal.forward( state );
    _qNetworkLocal.train();

    if ( static_cast<float>( rand() ) / RAND_MAX > eps )
    {
        auto actionIndex = Qvalues.argmax( 1 ).item<int>();
        auto actionIndices = _actionSpace.decodeActionIndex( actionIndex );
        return Action( actionIndices.first, actionIndices.second );
    }
    // greedy :
    int randomIndex = rand() % _numActions;
    auto actionIndices = _actionSpace.decodeActionIndex( randomIndex );
    return Action( actionIndices.first, actionIndices.second );
}


void Agent::learn( const std::vector<Experience> &experiences, const float gamma )
{
    std::vector<torch::Tensor> states, actions, next_states;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;

    for ( const Experience &experience : experiences )
    {
        states.push_back( experience.state.to( device ) );
        actions.push_back( experience.action.to( device ) );
        rewards.push_back( experience.reward );
        next_states.push_back( experience.nextState.to( device ) );
        dones.push_back( static_cast<uint8_t>( experience.done ) );
    }

    // Create tensors from vectors
    const auto statesTensor = torch::cat( states, 0 );
    const auto actionsTensor = torch::cat( actions, 0 );
    const auto rewardsTensor =
        torch::tensor( rewards, torch::dtype( torch::kFloat32 ) ).to( device );
    const auto nextStatesTensor = torch::cat( next_states, 0 );
    const auto doneTensor = torch::tensor( dones, torch::dtype( torch::kUInt8 ) ).to( device );

    // Network and loss computations
    const auto currentQValues = _qNetworkLocal.forward( nextStatesTensor ).detach().argmax( 1 );
    const auto QTargetsNext = _qNetworkTarget.forward( nextStatesTensor )
                                  .detach()
                                  .gather( 1, currentQValues.unsqueeze( -1 ) )
                                  .squeeze( -1 );
    const auto QTargets =
        rewardsTensor + gamma * QTargetsNext * ( 1 - doneTensor.to( torch::kFloat32 ) );
    const auto QExpected = _qNetworkLocal.forward( statesTensor )
                               .gather( 1, actionsTensor.unsqueeze( -1 ) )
                               .squeeze( -1 );
    const auto loss = torch::mse_loss( QExpected, QTargets );

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
