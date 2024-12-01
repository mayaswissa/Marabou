#include "DQNDelayedReplayBuffer.h"
DelayedReplayBuffer::DelayedReplayBuffer()
{
}

Experience& DelayedExperience::getExperience(){
    return _experience;
}

void DelayedReplayBuffer::addExperience( const torch::Tensor &state,
                                         const torch::Tensor &action,
                                         float reward,
                                         const torch::Tensor &nextState,
                                         bool done,
                                         unsigned depth,
                                         unsigned delay )
{
    auto experience = Experience( state, action, reward, nextState, done );
    auto delayExperience = DelayedExperience(  experience , depth, delay );
    _delayedExperience.append( delayExperience );
}

int DelayedReplayBuffer::getDepth() const
{
    if ( _delayedExperience.empty() )
        return -1;
    return _delayedExperience.last()._depth;
}

DelayedExperience DelayedReplayBuffer::popLast()
{
    return _delayedExperience.pop();
}

unsigned DelayedReplayBuffer::getSize() const
{
    return _delayedExperience.size();
}




