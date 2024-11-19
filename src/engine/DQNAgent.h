#ifndef DQNAGENT_H
#define DQNAGENT_H

#include "DQNActionSpace.h"
#include "DQNActoin.h"
#include "DQNNetwork.h"
#include "DQNReplayBuffer.h"
#undef Warning
#include <torch/torch.h>

class Agent
{
public:
    Agent( const ActionSpace &actionSpace );
    void step( const torch::Tensor &state,
               const torch::Tensor &action,
               unsigned reward,
               const torch::Tensor &nextState,
               bool done );
    Action act( const torch::Tensor &state, float eps = 0.1 );
    void learn( const std::vector<Experience> &experiences, float gamma );
    torch::Device getDevice() const;
    Agent( unsigned numVariables,
           unsigned numPhaseStatuses,
           unsigned embeddingDim,
           ActionSpace &actionSpace );
    Action tensorToAction( const torch::Tensor &tensor );

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    const ActionSpace &_actionSpace;
    unsigned _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions;
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