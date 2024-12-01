#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <random>

void Experience::updateReward( double newReward )
{
    this->reward = newReward;
    this->returned = true; // todo ?
}

ReplayBuffer::ReplayBuffer( const unsigned actionSize,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _actionSize( actionSize )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
    , _numExperiences( 0 )
    , _numReturnedExperiences( 0 )
{
}

void ReplayBuffer::add( const torch::Tensor &state,
                        const torch::Tensor &action,
                        float reward,
                        const torch::Tensor &nextState,
                        bool done,
                        unsigned depth,
                        unsigned numSplits )
{
    if ( _experiences.size() >= _bufferSize )
    {
        _experiences.popFirst();
        _numExperiences--;
        _numReturnedExperiences --;
    }


    auto experience = Experience( state, action, reward, nextState, done, depth, numSplits );
    printf("adding experience!\n");
    fflush(stdout);
    _experiences.append( experience );
    _numExperiences++;
}

Experience &ReplayBuffer::getExperienceAt( const unsigned index )
{
    printf("buffer 46\n");
    fflush(stdout);
    return _experiences[index];
}


Vector<unsigned> ReplayBuffer::sample() const
{
    printf("sample:\n _numReturnedExperiences %d\n", _numReturnedExperiences );
    fflush(stdout);
    Vector<unsigned> sampledIndices;

    if ( _experiences.empty() || _batchSize == 0 )
    {
        printf("experiences empty\n");
        fflush(stdout);
        return sampledIndices;
    }

    // todo sample from range of valid experiences
    int sampleSize = static_cast<int>( std::min( _batchSize, _numReturnedExperiences ) );
    Vector<unsigned> indices( _experiences.size() );
    std::iota( indices.begin(), indices.end(), 0 );
    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( indices.begin(), indices.end(), g );

    for ( int i = 0; i < sampleSize; ++i )
    {
        auto it = sampledIndices.end();
        sampledIndices.insert( it, indices[i] );
    }
    return sampledIndices;
}

unsigned ReplayBuffer::size() const
{
    return _numExperiences;
}

unsigned ReplayBuffer::getNumReturnedExperiences() const
{
    return _numReturnedExperiences;
}

void ReplayBuffer::updateReturnedWhenDoneSuccess()
{
    _numReturnedExperiences = _numExperiences;
}

void ReplayBuffer::increaseNumReturned()
{
    _numReturnedExperiences++;
}
