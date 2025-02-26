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
    void stepAlternativeAction( const State &stateBeforeSplit,
                                unsigned numSplits,
                                unsigned &numInconsistent,
                                double prunedSubtrees );
    void stepNewAction( const State &previousState,
                        const Action &action,
                        double reward,
                        const State &currentState,
                        bool done,
                        unsigned numSplits,
                        bool changeReward );

    void handleDone( const State &currentState, unsigned numSplits, double prunedSubtrees );
    std::unique_ptr<Action> act( const State &state, double eps = 0.1 );
    double updateLR();
    Action tensorToAction( const torch::Tensor &tensor ) const;
    void saveNetworks() const;
    void loadNetworks();
    int getActionStackSize() const;
    int getReplayBufferSize() const;
    void schedulersStep();

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    void learn();
    torch::Device getDevice() const;

    ActionSpace _actionSpace;
    unsigned _numPlConstraints, _numPhases, _numActions;
    unsigned _tStep;
    static constexpr double GAMMA = 0.9; // future rewards contribution to the current Q-value
    torch::Device device;
    const std::string _saveAgentFilePath;
    const std::string _trainedAgentFilePath;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam _optimizer;
    torch::optim::StepLR _scheduler;
    ReplayBuffer _replayedBuffer;
    bool handleInvalidGradients();
};
#endif