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
    State _previousState;
    Action _action;
    double _reward;
    State _currentState;
    bool _done;
    unsigned _depth;
    unsigned _numSplits;
    bool _changeReward;

    // Existing constructor
    Experience( State previousState,
                Action action,
                double reward,
                State currentState,
                const bool done,
                unsigned depth,
                unsigned numSplits = 0,
                bool changeReward = true )
        : _previousState( previousState )
        , _action( action )
        , _reward( reward )
        , _currentState( currentState )
        , _done( done )
        , _depth( depth )
        , _numSplits( numSplits )
        , _changeReward( changeReward )
    {
    }

    Experience( const Experience &other )
        : _previousState( other._previousState )
        , _action( other._action )
        , _reward( other._reward )
        , _currentState( other._currentState )
        , _done( other._done )
        , _depth( other._depth )
        , _numSplits( other._numSplits )
        , _changeReward( other._changeReward )
    {
    }

    Experience( Experience &&other ) noexcept
        : _previousState( std::move( other._previousState ) )
        , _action( std::move( other._action ) )
        , _reward( other._reward )
        , _currentState( std::move( other._currentState ) )
        , _done( other._done )
        , _depth( other._depth )
        , _numSplits( other._numSplits )
        , _changeReward( other._changeReward )
    {
    }

    Experience& operator=(Experience&& other) noexcept {
        if (this != &other) {
            _previousState = std::move(other._previousState);
            _action = std::move(other._action);
            _reward = other._reward;
            _currentState = std::move(other._currentState);
            _done = other._done;
            _depth = other._depth;
            _numSplits = other._numSplits;
            _changeReward = other._changeReward;
        }
        return *this;
    }
    void updateReward( double newReward );
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
    Experience &getRevisitExperienceAt( unsigned index );
    Experience &getExperienceAt( unsigned index );
    Vector<unsigned> sample() const;
    unsigned getNumExperiences() const;
    unsigned getNumRevisitExperiences() const;
    unsigned getExperienceBufferDepth() const;
    unsigned getBatchSize() const;
    void moveToRevisitExperiences();
    void addToRevisitExperiences( State state,
                                   Action action,
                                   double reward,
                                   State nextState,
                                   const bool done,
                                   unsigned depth,
                                   unsigned numSplits = 0,
                                   bool changeReward = true );

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    unsigned _numExperiences;
    unsigned _numRevisitedExperiences;
    unsigned _experienceBufferDepth;
    std::deque<std::unique_ptr<Experience>> _experiences;
    std::deque<std::unique_ptr<Experience>> _revisitExperiences;
};

#endif