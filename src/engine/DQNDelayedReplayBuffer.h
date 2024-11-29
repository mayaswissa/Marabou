#ifndef DQNDELAYEDREWARDBUFFER_H
#define DQNDELAYEDREWARDBUFFER_H
#include "DQNReplayBuffer.h"
#include "Vector.h"
struct DelayedExperience
{
    std::unique_ptr<Experience> _experience;
    unsigned _depth;
    unsigned _delay;
    DelayedExperience( std::unique_ptr<Experience> experience, const unsigned depth, const unsigned delay )
        : _experience( std::move(experience) )
        , _depth( depth )
        , _delay( delay )
    {
    }
};

class DelayedReplayBuffer
{
public:
    DelayedReplayBuffer();
    void addExperience( const torch::Tensor &state,
                        const torch::Tensor &action,
                        float reward,
                        const torch::Tensor &nextState,
                        bool done, unsigned depth, unsigned delay );

private:
    Vector<DelayedExperience> _actionsMemory;
};

#endif // DQNDELAYEDREWARDBUFFER_H
