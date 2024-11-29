#ifndef DQNDELAYEDREPLAYBUFFER_h
#define DQNDELAYEDREPLAYBUFFER_h
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
    std::unique_ptr<Experience> getExperience();
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
    unsigned getDepth() const;
    DelayedExperience popLast();
    unsigned getSize() const;
private:
    Vector<DelayedExperience> _actionsMemory;
};

#endif
