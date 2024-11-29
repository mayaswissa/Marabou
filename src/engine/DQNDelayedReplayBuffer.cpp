#include "DQNDelayedRewardBuffer.h"

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

