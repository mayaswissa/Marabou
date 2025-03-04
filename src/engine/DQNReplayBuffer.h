#ifndef DQNREPLAYBUFFER_H
#define DQNREPLAYBUFFER_H
#include "DQNActoin.h"
#include "DQNState.h"
#include "Vector.h"

#include <deque>
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
    double _soiScoreBeforeActiveAction;
    ActiveAction( const Action &action,
                  const State &stateBeforeAction,
                  unsigned splitsBeforeAction,
                  double soiScoreBeforeActiveAction )
        : _action( action )
        , _stateBeforeAction( stateBeforeAction )
        , _splitsBeforeActiveAction( splitsBeforeAction )
        , _soiScoreBeforeActiveAction( soiScoreBeforeActiveAction )
    {
    }
};

struct ActionEntry
{
    // pairs of actions and numSplits when act
    List<ActiveAction> _activeActions;
    List<Action> _alternativeActions;
    State _stateBeforeAction;

    ActionEntry( const Action &action,
                 const State &stateBeforeAction,
                 const unsigned splitsBeforeAction,
                 const double soiBeforeAction)
        : _stateBeforeAction( stateBeforeAction )

    {
        _activeActions = List<ActiveAction>();
        _activeActions.append( ActiveAction( action, stateBeforeAction, splitsBeforeAction, soiBeforeAction ) );
        _alternativeActions = List<Action>();

        auto const actionPhase = action.getActionPhase();
        ASSERT( actionPhase == RELU_PHASE_ACTIVE || actionPhase == RELU_PHASE_INACTIVE );
        const unsigned alternativeActionPhase =
            actionPhase == RELU_PHASE_ACTIVE ? RELU_PHASE_INACTIVE : RELU_PHASE_ACTIVE;
        const auto alternateAction = Action( action.getNumPhases(),
                                             action.getNumPlConstraints(),
                                             action.getActionPlConstraintIndex(),
                                             alternativeActionPhase );
        _alternativeActions.append( alternateAction );
    }
};

class ReplayBuffer
{
public:
    ReplayBuffer( unsigned numConstraints, unsigned bufferSize, unsigned batchSize );
    std::vector<unsigned> sample() const;
    unsigned getNumRevisitExperiences() const;
    unsigned getBatchSize() const;
    void addExperienceToRevisitBuffer( const State &state,
                                       const Action &action,
                                       double reward,
                                       const State &nextState,
                                       const bool done );

    bool compareStateWithAlternative( State &state ) const;

    void
    pushActionEntry( const Action &action,
                          const State &stateBeforeAction,
                          unsigned numSplitsBeforeAction,
                          double soiScoreBeforeAction );
    void handleDone( const State &currentState, unsigned numSplits, double soiScore );
    void moveActionToRevisitBuffer( const State &stateAfterAction,
                                    unsigned numSplitsAfterAction,
                                    ActionEntry *actionEntry,
                                    double soiScoreAfterAction );
    void applyNextAction( const State &state,
                          unsigned numSplits,
                          unsigned &numInconsistent,
                          double soiScore );
    int getActionStackSize() const;
    torch::Tensor getStates();
    torch::Tensor getActions();
    torch::Tensor getRewards();
    torch::Tensor getNextStates();
    torch::Tensor getDones();

private:
    unsigned _numConstraints;
    unsigned _bufferSize;
    unsigned _batchSize;
    List<ActionEntry *> _actionsStack;

    unsigned _size;            // valid entries in replayBuffer
    unsigned _writePosition;   //  pointer for the next empty position in experiences buffer
    torch::Tensor _states;     // [bufferSize, stateDim]
    torch::Tensor _actions;    // [bufferSize, 1]
    torch::Tensor _rewards;    // [bufferSize]
    torch::Tensor _nextStates; // [bufferSize, stateDim]
    torch::Tensor _dones;      // [bufferSize]
};

#endif