#include "Agent.h"
#include <random>
#include <utility>

Agent::Agent(int64_t stateSize, int64_t actionSize)
    : _stateSize(stateSize), _actionSize(actionSize),
      _qNetworkLocal(stateSize, actionSize),
      _qNetworkTarget(stateSize, actionSize),
      optimizer(_qNetworkLocal.parameters(), torch::optim::AdamOptions(LR)),
      memory(actionSize, 1e5, BATCH_SIZE),
      _tStep(0),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    _qNetworkLocal.to(device);
    _qNetworkTarget.to(device);
}

void Agent::step(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState, torch::Tensor done) {
    // save experience in replay memory
    memory.add(std::move(state), std::move(action), std::move(reward), std::move(nextState), std::move(done));
    _tStep = (_tStep + 1) % UPDATE_EVERY;
    if (_tStep == 0 && memory.size() > BATCH_SIZE) {
        const auto experiences = memory.sample();
        learn(experiences, GAMMA);
    }
}

int64_t Agent::act(torch::Tensor state, double eps) {
    state = state.to(device);
    _qNetworkLocal.eval();
    torch::NoGradGuard no_grad;
    const auto action_values = _qNetworkLocal.forward(state);
    _qNetworkLocal.train();
    if ( float randValue = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
         randValue > eps )
    {
        const int bestAction = action_values.argmax( 1 ).item<int>();
        return bestAction;
    }
    return rand() % _actionSize;
}


void Agent::learn(const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>& experiences, double gamma) {
    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for ( const auto &e : experiences )
    {
        states.push_back( std::get<0>( e ) );
        actions.push_back( std::get<1>( e ) );
        rewards.push_back( std::get<2>( e ) );
        next_states.push_back( std::get<3>( e ) );
        dones.push_back( std::get<4>( e ) );
    }
    const auto statesTensor = torch::cat( states, 0 ).to( device );
    const auto actionsTensor = torch::cat( actions, 0 ).to( device );
    const auto rewardsTensor = torch::cat( rewards, 0 ).to( device );
    const auto nextStatesTensor = torch::cat( next_states, 0 ).to( device );
    const auto doneTensor = torch::cat( dones, 0 ).to( device );
    const auto QTargetsNext =
        std::get<0>( _qNetworkTarget.forward( nextStatesTensor ).detach().max( 1 ) );
    const auto QTargets = rewardsTensor + ( gamma * QTargetsNext * ( 1 - doneTensor ) );
    const auto QExpected = _qNetworkLocal.forward( statesTensor ).gather( 1, actionsTensor );
    const auto loss = torch::mse_loss(QExpected, QTargets);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    softUpdate(_qNetworkLocal, _qNetworkTarget, TAU);
}

void Agent::softUpdate( const QNetwork & localModel, const QNetwork & targetModel, double tau) {
    const auto localParams = localModel.get_parameters();
    const auto targetParams = targetModel.get_parameters();
    for (size_t i = 0; i < localParams.size(); ++i) {
        targetParams[i].data().copy_(tau * localParams[i].data() + (1.0 - tau) * targetParams[i].data());
    }
}