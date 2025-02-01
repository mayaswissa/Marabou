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
    Agent( unsigned numPlConstraints,
           unsigned numPhases,
           const std::string &saveAgentPath,
           const std::string &trainedAgentPath = "" );
    void addAlternativeAction( const State &stateBeforeSplit,
                               unsigned depthBeforeSplit,
                               unsigned numSplits,
                               unsigned &numInconsistent,
                               double prunedSubtrees );
    void step( const State &state,
               const Action &action,
               double reward,
               const State &nextState,
               bool done,
               unsigned depth,
               unsigned numSplits,
               bool changeReward );

    void handleDone( const State &currentState,
                     unsigned stackDepth,
                     unsigned numSplits,
                     double prunedSubtrees );
    Action act( const State &state, double eps = 0.1 );
    double updateLR();
    Action tensorToAction( const torch::Tensor &tensor ) const;
    void saveNetworks() const;
    void loadNetworks();
    int getActionStackSize() const;
    int getReplayBufferSize() const;

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    void learn();
    torch::Device getDevice() const;

    ActionSpace _actionSpace;
    unsigned _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions;
    unsigned _tStep;
    static constexpr double GAMMA = 0.9;
    static constexpr double TAU = 1e-3; // Soft Update Parameter for target network
    static constexpr double LR = 1e-4;
    unsigned int learningSteps = 0;
    unsigned _updateEvery = 4;
    unsigned _batchSize = 100;
    torch::Device device;
    const std::string _saveAgentFilePath;
    const std::string _trainedAgentFilePath;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer _replayedBuffer;
    bool handleInvalidGradients();
};
#endif