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

    Experience( torch::Tensor state,
                const torch::Tensor &action,
                double reward,
                const torch::Tensor &nextState,
                const bool done )
        : state( std::move( state ) )
        , action( action )
        , reward( reward )
        , nextState( nextState )
        , done( done )
    {
    }
    void updateReward( double newReward );
};


class ReplayBuffer
{
public:
    ReplayBuffer( unsigned actionSize, unsigned bufferSize, unsigned batchSize );
    void add( const torch::Tensor &state,
              const torch::Tensor &action,
              float reward,
              const torch::Tensor &nextState,
              bool done );
    void add( std::unique_ptr<Experience> experience );

    Vector<std::unique_ptr<Experience>>  sample() const;
    size_t size() const;

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    Vector<std::unique_ptr<Experience>> _memory;
};

#endif