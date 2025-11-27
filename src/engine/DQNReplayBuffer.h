#ifndef DQNREPLAYBUFFER_H
#define DQNREPLAYBUFFER_H
#include "DQNAction.h"
#include "DQNState.h"

#include <utility>
#undef Warning
#include <torch/torch.h>

struct Experience
{
    State _stateBeforeAction;
    Action _action;
    double _reward;
    State _stateAfterAction;
    bool _done;
    unsigned _splitsBefore;
    bool _changeReward;

    Experience( const State &stateBeforeAction,
                const Action &action,
                const double reward,
                const State &stateAfterAction,
                const bool done,
                const unsigned numSplits = 0,
                const bool changeReward = true )
        : _stateBeforeAction( stateBeforeAction )
        , _action( action )
        , _reward( reward )
        , _stateAfterAction( stateAfterAction )
        , _done( done )
        , _splitsBefore( numSplits )
        , _changeReward( changeReward )
    {
    }

    Experience( const Experience &other )
        : _stateBeforeAction( other._stateBeforeAction )
        , _action( other._action )
        , _reward( other._reward )
        , _stateAfterAction( other._stateAfterAction )
        , _done( other._done )
        , _splitsBefore( other._splitsBefore )
        , _changeReward( other._changeReward )
    {
    }

    Experience( Experience &&other ) noexcept
        : _stateBeforeAction( std::move( other._stateBeforeAction ) )
        , _action( std::move( other._action ) )
        , _reward( other._reward )
        , _stateAfterAction( std::move( other._stateAfterAction ) )
        , _done( other._done )
        , _splitsBefore( other._splitsBefore )
        , _changeReward( other._changeReward )
    {
    }

    Experience &operator=( Experience &&other ) noexcept
    {
        if ( this != &other )
        {
            _stateBeforeAction = std::move( other._stateBeforeAction );
            _action = std::move( other._action );
            _reward = other._reward;
            _stateAfterAction = std::move( other._stateAfterAction );
            _done = other._done;
            _splitsBefore = other._splitsBefore;
            _changeReward = other._changeReward;
        }
        return *this;
    }
};

struct ActiveAction
{
    Action _action;
    State _stateBeforeAction;
    unsigned _splitsBeforeActiveAction;
    ActiveAction( const Action &action,
                  const State &stateBeforeAction,
                  unsigned splitsBeforeAction )
        : _action( action )
        , _stateBeforeAction( stateBeforeAction )
        , _splitsBeforeActiveAction( splitsBeforeAction )
    {
    }
};

struct ActionEntry
{
    List<ActiveAction> _activeActions;
    List<Action> _alternativeActions;
    State _stateBeforeAction;
    bool _isFake;
    bool _isDemo;
    bool _done;
    ActionEntry( const Action &action,
                 const State &stateBeforeAction,
                 const unsigned splitsBeforeAction,
                 const bool isFake,
                 const bool demo,
                 const bool done = false )
        : _stateBeforeAction( stateBeforeAction )
        , _isFake( isFake )
        , _isDemo( demo )
        , _done( done )
    {
        _activeActions = List<ActiveAction>();
        _activeActions.append( ActiveAction( action, stateBeforeAction, splitsBeforeAction ) );
        _alternativeActions = List<Action>();

        auto const actionPhase = action.getActionPhase();
        if (actionPhase !=DQN_RELU_ACTIVE && actionPhase !=DQN_RELU_INACTIVE){
        std ::cout <<"action phase not fixed ! action phase : " << actionPhase << std::endl;
        }
        ASSERT( actionPhase == DQN_RELU_ACTIVE || actionPhase == DQN_RELU_INACTIVE );
        const unsigned alternativeActionPhase =
            actionPhase == DQN_RELU_ACTIVE ? DQN_RELU_INACTIVE : DQN_RELU_ACTIVE;
        const auto alternateAction = Action( DQN_NUM_PHASES,
                                             action.getNumPlConstraints(),
                                             action.getActionPlConstraintIndex(),
                                             alternativeActionPhase );
        _alternativeActions.append( alternateAction );
    }
};

struct SampledBatch
{
    std::vector<unsigned> indices;
    std::vector<float> weights;
    std::vector<bool> isDemo;
};

class ReplayBuffer
{
public:
    ReplayBuffer( unsigned numConstraints, unsigned bufferSize, unsigned batchSize );
    void pushFakeActionEntry( const State &stateBeforeAction, unsigned numSplitsBeforeAction );
    unsigned getNumRevisitExperiences() const;
    unsigned getBatchSize() const;

    void pushActionEntry( const Action &action,
                          const State &stateBeforeAction,
                          unsigned numSplitsBeforeAction,
                          bool demo = false,
                          bool done = false );
    void handleDone( const State &currentState, unsigned numSplits );
    double potentialSubtreeSize() const;
    void moveActionToRevisitBuffer( const State &stateAfterAction,
                                    unsigned numSplitsAfterAction,
                                    ActionEntry *actionEntry,
                                    bool done = false );
    void applyNextAction( const State &state, unsigned numSplits, unsigned &numInconsistent );
    void addExperienceToRevisitBuffer( const State &state,
                                       const Action &action,
                                       double reward,
                                       const State &nextState,
                                       const bool done,
                                       bool isDemo = false );
    SampledBatch sample();
    void updatePriority( unsigned idx, float newP );
    int getActionStackSize() const;
    torch::Tensor getStates();
    torch::Tensor getActions();
    torch::Tensor getRewards();
    torch::Tensor getNextStates();
    torch::Tensor getDones();
    long getDemoFactor() const;
    long getEpsilonDemo() const;
    long getEpsilonAgent() const;
    long getEpsilon() const;


private:
    unsigned _numConstraints;
    unsigned _bufferSize;
    unsigned _batchSize;
    List<ActionEntry *> _actionsStack;
    unsigned _fakeActionIndex;
    unsigned _size;            // valid entries in replayBuffer
    unsigned _writePosition;   // pointer for the next empty position in experiences buffer
    torch::Tensor _states;     // [bufferSize, stateDim]
    torch::Tensor _actions;
    torch::Tensor _rewards;
    torch::Tensor _nextStates;
    torch::Tensor _dones;
    std::vector<float> _priorities;
    std::vector<float> _sumTree;
    std::vector<bool> _isDemo;
    float _maxPriority;
    float _demoFactor, _eps;
    float _beta, _betaInc;
    float _epsAgent = 1e-6f;
    float _epsDemo = 1.0f;
    void rebuildTree( unsigned ti );
};

#endif