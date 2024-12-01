#ifndef DQNREPLAYBUFFER_H
#define DQNREPLAYBUFFER_H
#include "Vector.h"

#include <deque>
#include <utility>
#undef Warning
#include <torch/torch.h>

struct Experience
{
    torch::Tensor state;
    torch::Tensor action;
    float reward;
    torch::Tensor nextState;
    bool done;
    unsigned depth;
    unsigned numSplits;
    bool returned;

    Experience( torch::Tensor state,
                const torch::Tensor &action,
                double reward,
                const torch::Tensor &nextState,
                const bool done,
                unsigned depth,
                unsigned numSplits = 0 )
        : state( std::move( state ) )
        , action( std::move( action ) )
        , reward( reward )
        , nextState( std::move( nextState ) )
        , done( done )
        , depth( depth )
        , numSplits( numSplits )
        , returned( false )
    {
    }
    void updateReward( double newReward );
    // unsigned getDepth() const;
};


class ReplayBuffer
{
public:
    ReplayBuffer( unsigned actionSize, unsigned bufferSize, unsigned batchSize );
    void add( const torch::Tensor &state,
              const torch::Tensor &action,
              float reward,
              const torch::Tensor &nextState,
              bool done,
              unsigned depth,
              unsigned numSplits = 0 );
    Experience &getExperienceAt( unsigned index );
    Vector<unsigned> sample() const;
    unsigned size() const;
    unsigned getNumReturnedExperiences() const;
    void updateReturnedWhenDoneSuccess();
    void increaseNumReturned();

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    // number of experiences in the buffer
    unsigned _numExperiences;
    // The number of experiences we revisited after completing a branch search in the search tree.
    unsigned _numReturnedExperiences;
    Vector<Experience> _experiences;
};

#endif