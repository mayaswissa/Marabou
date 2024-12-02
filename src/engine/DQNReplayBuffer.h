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
    float reward;
    State nextState;
    bool done;
    unsigned depth;
    unsigned numSplits;
    bool revisit;
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
        , revisit( false )
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
        , revisit( other.revisit )
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
        , revisit( other.revisit )
        , changeReward( other.changeReward )
    {
    }

    Experience &operator=( const Experience &other );
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
    Experience &getExperienceAt( unsigned index );
    Vector<unsigned> sample() const;
    unsigned size() const;
    int getNumRevisitExperiences() const;
    void updateReturnedWhenDoneSuccess();
    void increaseNumReturned();

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    // number of experiences in the buffer
    int _numExperiences;
    // The number of experiences we revisited after completing a branch search in the search tree.
    int _numRevisitExperiences;
    Vector<Experience> _experiences;
    Vector<Experience> _revisitExperiences;
};

#endif