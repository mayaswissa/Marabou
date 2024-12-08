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
    void step( State state,
               Action action,
               double reward,
               State nextState,
               bool done,
               unsigned depth,
               unsigned numSplits,
               bool changeReward );

    void handleDone( bool success );
    Action act( const torch::Tensor &state, double eps = 0.1 );
    Action tensorToAction( const torch::Tensor &tensor ) const;
    void saveNetworks() const;
    void loadNetworks();
    void moveRevisitExperience( unsigned currentNumSplits, unsigned depth, State *state );

private:
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );
    void learn();
    torch::Device getDevice() const;
    bool isEqualState( const State *state, const State *other ) const;
    void moveExperiencesToRevisitBuffer( unsigned currentNumSplits, unsigned depth, State *state );

    ActionSpace _actionSpace;
    unsigned _numPlConstraints, _numPhaseStatuses, _embeddingDim, _numActions;
    QNetwork _qNetworkLocal, _qNetworkTarget;
    torch::optim::Adam optimizer;
    ReplayBuffer _replayedBuffer;
    unsigned _tStep;
    static constexpr double GAMMA = 0.9;
    static constexpr double TAU = 1e-3;
    static constexpr double LR = 5e-4;
    static constexpr unsigned UPDATE_EVERY = 10; // todo change
    static constexpr unsigned BATCH_SIZE = 10;   // todo check
    torch::Device device;
    const std::string _saveAgentFilePath;
    const std::string _trainedAgentFilePath;
    bool handle_invalid_gradients();
};
#endif