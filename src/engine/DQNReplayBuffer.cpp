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
    , _numRevisitExperiences( 0 )
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

Experience &ReplayBuffer::getExperienceAt( const unsigned index )
{
    return _experiences[index];
}


Vector<unsigned> ReplayBuffer::sample() const
{
    printf( "sample:\n _numRevisitExperiences %d\n", getNumRevisitExperiences() );
    fflush( stdout );
    Vector<unsigned> sampledIndices;

    if ( _experiences.empty() || _batchSize == 0 || _numRevisitExperiences == 0 )
    {
        printf( "experiences empty\n" );
        fflush( stdout );
        return sampledIndices;
    }

    unsigned startIndex =  0;
    unsigned endIndex = numRevisitExperiences() - 1;

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

unsigned ReplayBuffer::numRevisitExperiences() const
{
    return _numRevisitExperiences;
}

void ReplayBuffer::decreaseNumRevisitExperiences()
{
    _numRevisitExperiences--;
}


int ReplayBuffer::getNumRevisitExperiences() const
{
    return _numRevisitExperiences;
}

void ReplayBuffer::updateReturnedWhenDoneSuccess()
{
    _numRevisitExperiences = _numExperiences;
}

void ReplayBuffer::increaseNumReturned()
{
    _numRevisitExperiences++;
}

void ReplayBuffer::moveToRevisitExperiences( const unsigned index )
{
    if ( _numRevisitExperiences >= _bufferSize )
    {
        auto experience = _revisitExperiences.popFirst();
        decreaseNumRevisitExperiences();
    }
    auto it = std::find_if( _experiences.begin(),
                            _experiences.end(),
                            [index]( const Experience &e ) { return e.depth == index; } );

    if ( it != _experiences.end() )
    {
        _revisitExperiences.append( std::move( *it ) );
        _numRevisitExperiences ++;
        _experiences.erase( it );
        _numExperiences--;
    }
}