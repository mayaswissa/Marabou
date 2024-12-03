#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <random>


Experience &Experience::operator=( const Experience &other )
{
    if ( this != &other )
    {
        state = other.state;
        action = other.action;
        reward = other.reward;
        nextState = other.nextState;
        done = other.done;
        depth = other.depth;
        numSplits = other.numSplits;
        changeReward = other.changeReward;
    }
    return *this;
}

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
    , _numExperiences( 0 )
    , _numRevisitedExperiences( 0 )
{
}

void ReplayBuffer::add( State state,
                        Action action,
                        double reward,
                        State nextState,
                        const bool done,
                        unsigned depth,
                        unsigned numSplits,
                        bool changeReward )
{
    if ( _experiences.size() >= _bufferSize )
    {
        auto experience = _experiences.pop();
        _numExperiences--;
    }

    auto experience =
        Experience( state, action, reward, nextState, done, depth, numSplits, changeReward );
    _experiences.append( experience );
    _numExperiences++;
}

void ReplayBuffer::addToRevisitExperiences( State state,
                        Action action,
                        double reward,
                        State nextState,
                        const bool done,
                        unsigned depth,
                        unsigned numSplits,
                        bool changeReward )
{
    if ( _revisitedExperiences.size() >= _bufferSize )
    {
        auto experience = _revisitedExperiences.pop();
        _numRevisitedExperiences--;
    }

    auto experience =
        Experience( state, action, reward, nextState, done, depth, numSplits, changeReward );
    _revisitedExperiences.append( experience );
    _numRevisitedExperiences++;
}

Experience &ReplayBuffer::getExperienceAt( const unsigned index )
{
    return _experiences[index];
}


Experience &ReplayBuffer::getRevisitedExperienceAt( const unsigned index )
{
    return _revisitedExperiences[index];
}


Vector<unsigned> ReplayBuffer::sample() const
{
    Vector<unsigned> sampledIndices;

    if ( _experiences.empty() || _batchSize == 0 || _numRevisitedExperiences == 0 )
    {
        printf( "experiences empty\n" );
        fflush( stdout );
        return sampledIndices;
    }

    unsigned startIndex =  0;
    unsigned endIndex = numRevisitedExperiences() - 1;

    unsigned rangeSize = endIndex - startIndex;
    unsigned sampleSize = std::min( _batchSize, rangeSize );

    Vector<unsigned> indices( rangeSize );
    std::iota( indices.begin(), indices.end(), startIndex );

    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( indices.begin(), indices.end(), g );

    for ( unsigned i = 0; i < sampleSize; ++i )
    {
        auto it = sampledIndices.end();
        sampledIndices.insert( it, indices[i] );
    }
    return sampledIndices;
}


unsigned ReplayBuffer::numExperiences() const
{
    return _numExperiences;
}

unsigned ReplayBuffer::numRevisitedExperiences() const
{
    return _numRevisitedExperiences;
}

void ReplayBuffer::decreaseNumRevisitExperiences()
{
    _numRevisitedExperiences--;
}


int ReplayBuffer::getNumRevisitExperiences() const
{
    return _numRevisitedExperiences;
}

void ReplayBuffer::updateReturnedWhenDoneSuccess()
{
    _numRevisitedExperiences = _numExperiences;
}

void ReplayBuffer::increaseNumReturned()
{
    _numRevisitedExperiences++;
}

void ReplayBuffer::moveToRevisitExperiences()
{
    if (_experiences.empty()) {
        // If there are no experiences, do nothing
        return;
    }

    // Check if the revisit experiences buffer is full
    if (_numRevisitedExperiences >= _bufferSize) {
        // Remove the first experience from revisit experiences if it's full
        _revisitedExperiences.popFirst();
        _numRevisitedExperiences--;
    }

    // Move the last experience from experiences to revisit experiences

    _revisitedExperiences.append(std::move(_experiences[_numExperiences - 1]));
    _numRevisitedExperiences++;
    _experiences.pop();
    _numExperiences--;
}
