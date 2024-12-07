#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <memory>
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
    , _numExperiences( 0 )
    , _numRevisitedExperiences( 0 )
    , _experienceBufferDepth( 0 )
    // , _experiences( bufferSize )
    // , _revisitedExperiences( bufferSize )
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
    if ( _numExperiences >= _bufferSize )
    {
        _experiences.pop_front();
        _numExperiences--;
    }

    auto experience = std::make_unique<Experience>( std::move( state ),
                                                    std::move( action ),
                                                    reward,
                                                    std::move( nextState ),
                                                    done,
                                                    depth,
                                                    numSplits,
                                                    changeReward );
    _experiences.push_back( std::move( experience ) );
    _experienceBufferDepth = depth;
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
    if ( getNumRevisitedExperiences() >= _bufferSize )
    {
        _revisitedExperiences.pop_front();
        _numRevisitedExperiences--;
    }

    auto experience = std::make_unique<Experience>( std::move( state ),
                                                    std::move( action ),
                                                    reward,
                                                    std::move( nextState ),
                                                    done,
                                                    depth,
                                                    numSplits,
                                                    changeReward );
    _revisitedExperiences.push_back( std::move( experience ) );
    _numRevisitedExperiences++;
}

Experience &ReplayBuffer::getExperienceAt( const unsigned index )
{
    if ( index >= _numExperiences )
        throw std::out_of_range( "Index out of range in experiences" ); // todo error
    return *_experiences[index];
}


Experience &ReplayBuffer::getRevisitedExperienceAt( const unsigned index )
{
    if ( index >= getNumRevisitedExperiences() )
        throw std::out_of_range( "Index out of range in revisited experiences" ); // todo error
    return *_revisitedExperiences[index];
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

    unsigned startIndex = 0;
    unsigned endIndex = getNumRevisitedExperiences() - 1;

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


unsigned ReplayBuffer::getNumExperiences() const
{
    return _numExperiences;
}

unsigned ReplayBuffer::getNumRevisitedExperiences() const
{
    return _numRevisitedExperiences;
}

unsigned ReplayBuffer::getExperienceBufferDepth() const
{
    return _experienceBufferDepth;
}

void ReplayBuffer::decreaseNumRevisitExperiences()
{
    _numRevisitedExperiences--;
}

void ReplayBuffer::increaseNumReturned()
{
    _numRevisitedExperiences++;
}

void ReplayBuffer::moveToRevisitExperiences()
{
    if ( _experiences.empty() )
    {
        return;
    }

    // Check if the revisit experiences buffer is full
    if ( _numRevisitedExperiences >= _bufferSize )
    {
        // Remove the first experience from revisit experiences if it's full
        _revisitedExperiences.pop_front();
        _numRevisitedExperiences--;
    }

    auto &experience = _experiences[_numExperiences - 1];
    _revisitedExperiences.push_back( std::move( experience ) );
    _numRevisitedExperiences++;
    _experiences.pop_back();
    _numExperiences--;
    if ( _numExperiences == 0 )
        _experienceBufferDepth = 0;
    else
        _experienceBufferDepth = _experiences.at( _numExperiences - 1 ).get()->depth;
}

unsigned ReplayBuffer::getBatchSize() const
{
    return _batchSize;
}