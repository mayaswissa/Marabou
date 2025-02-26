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
    State _stateAfterAction;
    unsigned _splitsBeforeActiveAction;
    ActiveAction( const Action &action,
                  const State &stateBeforeAction,
                  const State &stateAfterAction,
                  unsigned splitsBeforeAction )
        : _action( action )
        , _stateBeforeAction( stateBeforeAction )
        , _stateAfterAction( stateAfterAction )
        , _splitsBeforeActiveAction( splitsBeforeAction )
    {
    }
};

struct ActionsStack
{
    // pairs of actions and numSplits when act
    List<ActiveAction> _activeActions;
    List<Action> _alternativeActions;
    State _stateBeforeAction;

    ActionsStack( const Action &action,
                  const State &stateBeforeAction,
                  const State &stateAfterAction,
                  const unsigned splitsBeforeAction )
        : _stateBeforeAction( stateBeforeAction )

    {
        _activeActions = List<ActiveAction>();
        _activeActions.append( ActiveAction( action,
                                             stateBeforeAction ,
                                             stateAfterAction,
                                             splitsBeforeAction ) );
        _alternativeActions = List<Action>();

        const unsigned actionPhase = action.getAssignmentIndex() == 2 ? 1 : 2;
        const auto alternateAction =
            Action( action.getNumPhases(), action.getNumPlConstraints(), action.getPlConstraintActionIndex(), actionPhase );
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

    void pushActionEntry( const Action &action,
                          const State &stateBeforeAction,
                          const State &stateAfterAction,
                          unsigned numSplits );
    void handleDone( const State &currentState, unsigned numSplits, double prunedSubtrees );
    void moveActionToRevisitBuffer( const State &stateAfterAction,
                                    unsigned numSplits,
                                    ActionsStack *actionEntry,
                                    double prunedSubtrees );
    void applyNextAction( const State &state,
                          unsigned numSplits,
                          unsigned &numInconsistent,
                          double prunedSubtrees );
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
    std::deque<std::unique_ptr<Experience>> _revisitExperiences;
    List<ActionsStack *> _actionsStack;

    unsigned _size; // valid entries in replayBuffer
    unsigned _writePosition; //  pointer for the next empty position in experiences buffer
    torch::Tensor _states; // [bufferSize, stateDim]
    torch::Tensor _actions; // [bufferSize]
    torch::Tensor _rewards; // [bufferSize]
    torch::Tensor _nextStates; // [bufferSize, stateDim]
    torch::Tensor _dones; // [bufferSize]
};

#endif