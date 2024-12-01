#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <random>

void Experience::updateReward( double newReward )
{
    this->reward = newReward;
}

ReplayBuffer::ReplayBuffer( const unsigned actionSize,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _actionSize( actionSize )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
{
}

void ReplayBuffer::add( const torch::Tensor &state,
                        const torch::Tensor &action,
                        float reward,
                        const torch::Tensor &nextState,
                        bool done )
{
    if ( _experiences.size() >= _bufferSize )
        _experiences.eraseAt(0);


    auto experience =
        Experience( state, action, reward, nextState, done );
    _experiences.append( experience );
}
Experience ReplayBuffer::getExperienceAt( const unsigned index ) const
{
    return _experiences.get( index );
}
void ReplayBuffer::add( const Experience &experience )
{
    if ( _experiences.size() >= _bufferSize )
        _experiences.eraseAt(0);
    _experiences.append(experience);
}


Vector<unsigned> ReplayBuffer::sample() const
{
    Vector<unsigned> sampledIndices;

    if ( _experiences.empty() || _batchSize == 0 )
    {
        return sampledIndices;
    }

    int sampleSize = static_cast<int>(std::min( _batchSize, _experiences.size() ));
    Vector<unsigned> indices(_experiences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( indices.begin(), indices.end(), g );

    for (int i = 0; i < sampleSize; ++i) {
        auto it = sampledIndices.end();
        sampledIndices.insert(it, indices[i]);
    }
    return sampledIndices;
}

unsigned ReplayBuffer::size() const
{
    return _experiences.size();
}