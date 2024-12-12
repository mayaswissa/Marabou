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
    unsigned _depthAfter;
    unsigned _splitsBefore;
    bool _changeReward;

    // Existing constructor
    Experience( State stateBeforeAction,
                Action action,
                double reward,
                State stateAfterAction,
                const bool done,
                unsigned depth,
                unsigned numSplits = 0,
                bool changeReward = true )
        : _stateBeforeAction( stateBeforeAction )
        , _action( action )
        , _reward( reward )
        , _stateAfterAction( stateAfterAction )
        , _done( done )
        , _depthAfter( depth )
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
        , _depthAfter( other._depthAfter )
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
        , _depthAfter( other._depthAfter )
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
            _depthAfter = other._depthAfter;
            _splitsBefore = other._splitsBefore;
            _changeReward = other._changeReward;
        }
        return *this;
    }
    void updateReward( double newReward );
};

struct ActiveAction
{
    Action _action;
    State _stateBeforeAction;
    State _stateAfterAction;
    unsigned _depthBeforeAction;
    unsigned _splitsBeforeActiveAction;
    ActiveAction( Action action,
                  State stateBeforeAction,
                  State stateAfterAction,
                  unsigned depthBeforeAction,
                  unsigned splitsBeforeAction )
        : _action( action )
        , _stateBeforeAction( stateBeforeAction )
        , _stateAfterAction( stateAfterAction )
        , _depthBeforeAction( depthBeforeAction )
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

    ActionsStack( Action action,
                  State stateBeforeAction,
                  State stateAfterAction,
                  unsigned depthBeforeAction,
                  unsigned splitsBeforeAction ) : _stateBeforeAction( stateBeforeAction )

    {
        _activeActions = List<ActiveAction>();
        _activeActions.append( ActiveAction( action,
                                             std::move( stateBeforeAction ),
                                             std::move( stateAfterAction ),
                                             depthBeforeAction,
                                             splitsBeforeAction ) );
        _alternativeActions = List<Action>();

        unsigned actionPhase = action.getAssignmentIndex() == 2 ? 1 : 2;
        auto alternateAction =
            Action( action.getNumPhases(), action.getPlConstraintActionIndex(), actionPhase );
        _alternativeActions.append( alternateAction );
    }
};

class ReplayBuffer
{
public:
    ReplayBuffer( unsigned actionSize, unsigned bufferSize, unsigned batchSize );
    void add( State state,
              Action action,
              double reward,
              State nextState,
              const bool done,
              unsigned depth,
              unsigned numSplits = 0,
              bool changeReward = true );
    Experience &getRevisitExperienceAt( unsigned index ) const;
    Vector<unsigned> sample() const;
    unsigned getNumRevisitExperiences() const;
    unsigned getBatchSize() const;
    void addToRevisitExperiences( State state,
                                  Action action,
                                  double reward,
                                  State nextState,
                                  const bool done,
                                  unsigned depth,
                                  unsigned numSplits = 0,
                                  bool changeReward = true );

    bool compareStateWithAlternative( State &state ) const;

    void pushActionEntry( Action action,
                          State stateBeforeAction,
                          State stateAfterAction,
                          unsigned depth,
                          unsigned numSplits );
    void handleDone( State currentState, bool success, unsigned stackDepth, unsigned numSplits );
    void applyNextAction( State state, unsigned depth, unsigned numSplits );

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    unsigned _numRevisitedExperiences;
    std::deque<std::unique_ptr<Experience>> _revisitExperiences;
    void goToCurrentDepth( unsigned currentDepth );
    List<ActionsStack *> _actionsStack;
};

#endif