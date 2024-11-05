#ifndef AGENT_H
#define AGENT_H

#include "DQNReplayBuffer.h"
#include "DQNNetwork.h"

#undef Warning
#include <torch/torch.h>

class Agent {
public:
    Agent(unsigned stateSize, unsigned actionSize,
          unsigned hiddenLayer1Size, unsigned hiddenLayer2Size);
    void step(const torch::Tensor& state, const torch::Tensor &action,
             unsigned reward, const torch::Tensor& nextState,  bool done);
    torch::Tensor act(const torch::Tensor &state, float eps = 0.1);
    void learn(const std::vector<Experience>& experiences, const float gamma);
    torch::Device getDevice() const;
private:
    static void softUpdate( const QNetwork & localModel, const QNetwork & targetModel);

    unsigned _stateSize, _actionSize;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer _memory;
    unsigned _tStep;
    static constexpr float GAMMA = 0.99;
    static constexpr float TAU = 1e-3;
    static constexpr float LR = 5e-4;
    static constexpr unsigned UPDATE_EVERY = 4;
    static constexpr unsigned BATCH_SIZE = 10;
    torch::Device device;
};
#endif