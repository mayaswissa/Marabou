#ifndef DQNAGENT_H
#define DQNAGENT_H

#include "DQNActionSpace.h"
#include "DQNActoin.h"
#include "DQNNetwork.h"
#include "DQNReplayBuffer.h"
#include "DQNState.h"
#undef Warning
#include <torch/torch.h>

class Agent
{
public:
    Agent( unsigned numPlConstraints, unsigned numPhases, const std::string &saveAgentPath, const std::string &trainedAgentPath = "" );
    Agent( unsigned numVariables,
           unsigned numPhaseStatuses,
           unsigned embeddingDim,
           std::unique_ptr< ActionSpace > actionSpace );
    void step();
    void addToExperiences( unsigned currentNumSplits, State state,
                           Action action,
                           double reward,
                           State nextState,
                           const bool done,
                           unsigned depth,
                           unsigned numSplits,
                           bool changeReward );
    void saveNetworks() const;
    void loadNetworks();
    void handleDone( bool success );
    void moveExperiencesToRevisitedBuffer( unsigned currentNumSplits, unsigned depth, State* state );
    Action act( const torch::Tensor &state, double eps = 0.1 );
    void learn( const double gamma );
    torch::Device getDevice() const;
    Action tensorToAction( const torch::Tensor &tensor );
  unsigned getNumExperiences() const;

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    bool handle_invalid_gradients();
    ActionSpace _actionSpace;
    unsigned _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer _replayedBuffer;
    unsigned _tStep;
    unsigned _experienceDepth;
    static constexpr double GAMMA = 0.99;
    static constexpr double TAU = 1e-3;
    static constexpr double LR = 5e-4;
    static constexpr unsigned UPDATE_EVERY = 4; // todo change
    static constexpr unsigned BATCH_SIZE = 10;  // todo check
    torch::Device device;
    const std::string _saveAgentFilePath;
    const std::string _trainedAgentFilePath;
};
#endif