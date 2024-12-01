#ifndef DQNDELAYEDREPLAYBUFFER_H
#define DQNDELAYEDREPLAYBUFFER_H
#include "DQNReplayBuffer.h"
#include "Vector.h"
struct DelayedExperience
{
    Experience _experience;
    unsigned _depth;
    unsigned _delay;
    DelayedExperience( Experience &experience, const unsigned depth, const unsigned delay )
        : _experience( experience )
        , _depth( depth )
        , _delay( delay )
    {
    }
    Experience& getExperience();
};

class DelayedReplayBuffer
{
public:
    DelayedReplayBuffer();
    void addExperience( const torch::Tensor &state,
                        const torch::Tensor &action,
                        float reward,
                        const torch::Tensor &nextState,
                        bool done,
                        unsigned depth,
                        unsigned delay );
    int getDepth() const;
    DelayedExperience popLast();
    unsigned getSize() const;
    auto begin()
    {
        return _delayedExperience.begin();
    }
    auto end()
    {
        return _delayedExperience.end();
    }

private:
    Vector<DelayedExperience> _delayedExperience;
};

#endif
