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
    State state;
    Action action;
    double reward;
    State nextState;
    bool done;
    unsigned depth;
    unsigned numSplits;
    bool changeReward;

    // Existing constructor
    Experience( State state,
                Action action,
                double reward,
                State nextState,
                const bool done,
                unsigned depth,
                unsigned numSplits = 0,
                bool changeReward = true )
        : state( state )
        , action( action )
        , reward( reward )
        , nextState( nextState )
        , done( done )
        , depth( depth )
        , numSplits( numSplits )
        , changeReward( changeReward )
    {
    }

    Experience( const Experience &other )
        : state( other.state )
        , action( other.action )
        , reward( other.reward )
        , nextState( other.nextState )
        , done( other.done )
        , depth( other.depth )
        , numSplits( other.numSplits )
        , changeReward( other.changeReward )
    {
    }

    Experience( Experience &&other ) noexcept
        : state( std::move( other.state ) )
        , action( std::move( other.action ) )
        , reward( other.reward )
        , nextState( std::move( other.nextState ) )
        , done( other.done )
        , depth( other.depth )
        , numSplits( other.numSplits )
        , changeReward( other.changeReward )
    {
    }

    Experience& operator=(Experience&& other) noexcept {
        if (this != &other) {
            state = std::move(other.state);
            action = std::move(other.action);
            reward = other.reward;
            nextState = std::move(other.nextState);
            done = other.done;
            depth = other.depth;
            numSplits = other.numSplits;
            changeReward = other.changeReward;
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
    Experience &getRevisitedExperienceAt( unsigned index );
    Experience &getExperienceAt( unsigned index );
    Vector<unsigned> sample() const;
    unsigned getNumExperiences() const;
    unsigned getNumRevisitedExperiences() const;
    unsigned getExperienceBufferDepth() const;
    unsigned getBatchSize() const;
    void increaseNumReturned();
    void decreaseNumRevisitExperiences();
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
    // number of experiences in the buffer
    unsigned _numExperiences;
    // The number of experiences we revisited after completing a branch search in the search tree.
    unsigned _numRevisitedExperiences;

    unsigned _experienceBufferDepth;
    std::deque<std::unique_ptr<Experience>> _experiences;
    std::deque<std::unique_ptr<Experience>> _revisitedExperiences;
};

#endif