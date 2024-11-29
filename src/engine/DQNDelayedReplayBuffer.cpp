#include "DQNDelayedReplayBuffer.h"

std::unique_ptr<Experience> DelayedExperience::getExperience() {
    return std::move(_experience);
}

void DelayedReplayBuffer::addExperience( const torch::Tensor &state,
                                         const torch::Tensor &action,
                                         float reward,
                                         const torch::Tensor &nextState,
                                         bool done,
                                         unsigned depth,
                                         unsigned delay )
{
    auto experience = std::make_unique<Experience>( state, action, reward, nextState, done );
    DelayedExperience( std::move( experience ), depth, delay );
    _actionsMemory.append( DelayedExperience( std::move( experience ), depth, delay ) );
}

unsigned DelayedReplayBuffer::getDepth() const
{
    return _actionsMemory.last()._depth;
}

DelayedExperience DelayedReplayBuffer::popLast()
{
    return _actionsMemory.pop();
}

unsigned DelayedReplayBuffer::getSize() const
{
    return _actionsMemory.size();
}




