#include "DQNAgent.h"
#include <random>
#include <utility>

Agent::Agent(const unsigned stateSize, const unsigned actionSize,
             const unsigned hiddenLayer1Size, const unsigned hiddenLayer2Size)
    : _stateSize(stateSize), _actionSize(actionSize),
      _qNetworkLocal(stateSize, actionSize, hiddenLayer1Size, hiddenLayer2Size),
      _qNetworkTarget(stateSize, actionSize, hiddenLayer1Size, hiddenLayer2Size),
      optimizer(_qNetworkLocal.parameters(), torch::optim::AdamOptions(LR)),
      _memory(actionSize, 1e5, BATCH_SIZE),
      _tStep(0),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    _qNetworkLocal.to(device);
    _qNetworkTarget.to(device);
}

void Agent::step(const torch::Tensor& state, const torch::Tensor &action,
            const unsigned reward, const torch::Tensor& nextState, const bool done) {
    // save experience in replay memory
    _memory.add(state, action, reward, nextState, done);
    _tStep = (_tStep + 1) % UPDATE_EVERY;
    if (_tStep == 0 && _memory.size() > BATCH_SIZE) {
        const auto experiences = _memory.sample();
        learn(experiences, GAMMA);
    }
}

torch::Tensor Agent::act(const torch::Tensor &state, const float eps) {
    if (state.dim() != 2 || state.size(1) != _stateSize) {
        std::cerr << "State tensor dimensions incorrect in Agent::act, received dimensions: " << state.sizes() << std::endl;
        throw std::runtime_error("Incorrect state dimensions passed to Agent::act");
    }
    _qNetworkLocal.eval();
    const auto action_values = _qNetworkLocal.forward(state);
    _qNetworkLocal.train();

    // epsilon greedy:
    if (const float randValue = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
         randValue > eps )
    {
        return action_values.argmax( 1 );
    }
    return action_values[rand() % _actionSize];
}


void Agent::learn(const std::vector<Experience>& experiences, const float gamma) {
    // Using vectors to hold tensors and simple types
    std::vector<torch::Tensor> states, actions, next_states;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;

    for (const Experience& experience : experiences) {
        states.push_back(experience.state.to(device));
        actions.push_back(experience.action.to(device));
        rewards.push_back(experience.reward);
        next_states.push_back(experience.nextState.to(device));
        dones.push_back(static_cast<uint8_t>(experience.done));
    }

    // Create tensors from vectors
    const auto statesTensor = torch::cat(states, 0);
    const auto actionsTensor = torch::cat(actions, 0);
    const auto rewardsTensor = torch::tensor(rewards, torch::dtype(torch::kFloat32)).to(device);
    const auto nextStatesTensor = torch::cat(next_states, 0);
    const auto doneTensor = torch::tensor(dones, torch::dtype(torch::kUInt8)).to(device);

    // Network and loss computations
    const auto nextActions = _qNetworkLocal.forward(nextStatesTensor).detach().argmax(1);
    const auto QTargetsNext = _qNetworkTarget.forward(nextStatesTensor).detach().gather(1, nextActions.unsqueeze(-1)).squeeze(-1);
    const auto QTargets = rewardsTensor + gamma * QTargetsNext * (1 - doneTensor.to(torch::kFloat32));
    const auto QExpected = _qNetworkLocal.forward(statesTensor).gather(1, actionsTensor.unsqueeze(-1)).squeeze(-1);
    const auto loss = torch::mse_loss(QExpected, QTargets);

    // Backpropagation
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    softUpdate(_qNetworkLocal, _qNetworkTarget);
}


void Agent::softUpdate( const QNetwork & localModel, const QNetwork & targetModel) {
    const auto localParams = localModel.get_parameters();
    const auto targetParams = targetModel.get_parameters();
    for (size_t i = 0; i < localParams.size(); ++i) {
        targetParams[i].data().copy_(TAU * localParams[i].data() + (1.0 - TAU) * targetParams[i].data());
    }
}

torch::Device Agent::getDevice() const {
    return device;
}
