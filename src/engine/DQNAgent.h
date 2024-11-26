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
               double reward,
               const torch::Tensor &nextState,
               bool done,
               const bool run );
    Agent( unsigned numVariables,
           unsigned numPhaseStatuses,
           unsigned embeddingDim,
           ActionSpace &actionSpace );
    void saveNetworks( const std::string &filepath ) const;
    void loadNetworks();
    Action act( const torch::Tensor &state, double eps = 0.1 );
    void learn( const std::vector<Experience> &experiences, double gamma );
    torch::Device getDevice() const;
    Action tensorToAction( const torch::Tensor &tensor );

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    bool handle_invalid_gradients();
    const ActionSpace &_actionSpace;
    unsigned _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer _memory;
    unsigned _tStep;
    static constexpr double GAMMA = 0.99;
    static constexpr double TAU = 1e-3;
    static constexpr double LR = 5e-4;
    static constexpr unsigned UPDATE_EVERY = 4; // todo change
    static constexpr unsigned BATCH_SIZE = 10;  // todo check
    torch::Device device;
    const std::string _filePath;
};
#endif