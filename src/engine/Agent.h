#ifndef AGENT_H
#define AGENT_H

#include "ReplayBuffer.h"
#include "QNetwork.h"

#undef Warning
#include <torch/torch.h>

class Agent {
public:
    Agent(int64_t stateSize, int64_t actionSize);
    void step(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState, torch::Tensor done);
    int64_t act(torch::Tensor state, double eps = 0.1);
    void learn(const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>& experiences, double gamma);

private:
    static void softUpdate( const QNetwork & localModel, const QNetwork & targetModel, double tau);

    int64_t _stateSize, _actionSize;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer memory;
    int64_t _tStep;

    static constexpr double GAMMA = 0.99;
    static constexpr double TAU = 1e-3;
    static constexpr double LR = 5e-4;
    static constexpr int64_t UPDATE_EVERY = 4;
    static constexpr int64_t BATCH_SIZE = 64;
    torch::Device device;
};
#endif