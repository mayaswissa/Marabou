#ifndef DQNAGENT_H
#define DQNAGENT_H

#include "DQNAction.h"
#include "DQNActionSpace.h"
#include "DQNNetwork.h"
#include "DQNReplayBuffer.h"
#include "DQNState.h"

#undef Warning
#include <memory>
#include <string>
#include <torch/torch.h>

// Keep your logging macro style
#define DQN_LOG( x, ... ) MARABOU_LOG( GlobalConfiguration::DQN_LOGGING, "DQN: %s\n", x )

class Agent
{
public:
    Agent( unsigned numPlConstraints,
           unsigned numPhases,
           const std::string &trainedAgentPath = "" );

    // persistence
    void saveNetworks( const std::string &path ) const;
    void loadNetworks();

    // environment hooks
    void handleDone( const State &currentState, unsigned numSplits );
    void stepAlternativeAction( const State &stateBeforeSplit,
                                unsigned numSplits,
                                unsigned &numInconsistent );
    void stepFakeAction( const State &stateBeforeAction, unsigned numSplitsBeforeAction );
    void stepNewAction( const State &previousState,
                        const Action &action,
                        bool done,
                        unsigned numSplits,
                        bool isDemo = false ); // default ONLY in header

    // behavior
    std::unique_ptr<Action> actBestAction( const State &state );
    std::unique_ptr<Action> actRandomly( const State &state );

    // learning
    void learn();
    void schedulersStep();

    // misc
    torch::Device getDevice() const;
    int getActionStackSize() const;
    int getReplayBufferSize() const;

private:
    // target network update (EMA with tau)
    static void softUpdate( const QNetwork &localModel, const QNetwork &targetModel );

    // stability helpers
    bool handleInvalidGradients();

    // masks illegal choices (fixed rows or NOT_FIXED phase).
    // returns [B] bool tensor indicating terminal-by-mask rows (no legal action).
    torch::Tensor maskQInPlace( const torch::Tensor &state, torch::Tensor &Q ) const;

    // kept for compatibility with existing call sites; forwards to maskQInPlace.
    inline void applyActionMask( const torch::Tensor &tensorState, torch::Tensor &QValues ) const
    {
        (void)maskQInPlace( tensorState, QValues );
    }

private:
    // config / constants
    static constexpr double GAMMA = 0.9; // discount for next-state value

    // state
    ActionSpace _actionSpace;
    unsigned _numPlConstraints;
    unsigned _numPhases;
    unsigned _numActions;
    unsigned _tStep;

    // devices
    torch::Device device;

    // persistence
    const std::string _trainedAgentFilePath;

    // networks / opt
    QNetwork _qNetworkLocal;
    QNetwork _qNetworkTarget;
    torch::optim::Adam _optimizer;
    torch::optim::StepLR _scheduler;

    // replay / logging
    ReplayBuffer _replayedBuffer;
    unsigned _lossVerbosity;

    // DQfD / loss knobs
    float _lambdaSup;
    float _lambdaDecay; // multiplicative decay factor (computed in ctor)
    float _margin;
};

#endif // DQNAGENT_H
