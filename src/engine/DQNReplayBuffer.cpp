#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <memory>
#include <random>


void Experience::updateReward( double newReward )
{
    this->_reward = newReward;
}

ReplayBuffer::ReplayBuffer( const unsigned actionSize,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _actionSize( actionSize )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
    , _numExperiences( 0 )
    , _numRevisitedExperiences( 0 )
    , _numAlternativeSplits( 0 )
    , _experienceBufferDepth( 0 )
{
}

void ReplayBuffer::add( State previousState,
                        Action action,
                        double reward,
                        State currentState,
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

    auto experience = std::make_unique<Experience>( std::move( previousState ),
                                                    std::move( action ),
                                                    reward,
                                                    std::move( currentState ),
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
    if ( getNumRevisitExperiences() >= _bufferSize )
    {
        _revisitExperiences.pop_front();
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
    _revisitExperiences.push_back( std::move( experience ) );
    _numRevisitedExperiences++;
}

Experience &ReplayBuffer::getExperienceAt( const unsigned index )
{
    if ( index >= _numExperiences )
        throw std::out_of_range( "Index out of range in experiences" ); // todo error
    return *_experiences[index];
}


Experience &ReplayBuffer::getRevisitExperienceAt( const unsigned index )
{
    if ( index >= getNumRevisitExperiences() )
        throw std::out_of_range( "Index out of range in revisited experiences" ); // todo error
    return *_revisitExperiences[index];
}

void ReplayBuffer::setBufferDepth( const unsigned depth )
{
    _experienceBufferDepth = depth;
}

Vector<unsigned> ReplayBuffer::sample() const
{
    Vector<unsigned> sampledIndices;

    if ( _batchSize == 0 || _numRevisitedExperiences == 0 )
    {
        printf( "revisit experiences empty\n" );
        fflush( stdout );
        return sampledIndices;
    }

    unsigned startIndex = 0;
    unsigned endIndex = getNumRevisitExperiences() - 1;

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

unsigned ReplayBuffer::getNumRevisitExperiences() const
{
    return _numRevisitedExperiences;
}

unsigned ReplayBuffer::getExperienceBufferDepth() const
{
    return _experienceBufferDepth;
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
        _revisitExperiences.pop_front();
        _numRevisitedExperiences--;
    }

    auto &experience = _experiences[_numExperiences - 1];
    _revisitExperiences.push_back( std::move( experience ) );
    _numRevisitedExperiences++;
    _experiences.pop_back();
    _numExperiences--;
    if ( _numExperiences == 0 )
        _experienceBufferDepth = 0;
    else
        _experienceBufferDepth = _experiences.at( _numExperiences - 1 ).get()->_depth;
}

unsigned ReplayBuffer::getBatchSize() const
{
    return _batchSize;
}

void ReplayBuffer::addAlternativeAction( Action action,
                                         unsigned plConstraint,
                                         unsigned constraintPhase,
                                         State currentState,
                                         unsigned depth )
{
    auto alternativeSplit = std::make_unique<AlternativeSplits>(
        std::move( action ), plConstraint, constraintPhase, currentState, depth );
    _alternativeExperiences.push_back( std::move( alternativeSplit ) );
    _numAlternativeSplits++;
}

void ReplayBuffer::moveAlternativeActionToExperience( State stateBeforeSplit,
                                                      State stateAfterSplit,
                                                      unsigned numSplits )
{
    const auto &alternativeSplit = _alternativeExperiences[_numExperiences - 1];
    double reward = 0;

    add( std::move( stateBeforeSplit ),
         std::move( std::move( alternativeSplit.get()->action ) ),
         reward,
         std::move( stateAfterSplit ),
         false,
         alternativeSplit.get()->depthBeforeSplit + 1, // todo check depth
         numSplits,
         true );
    _alternativeExperiences.pop_back();
    _numAlternativeSplits--;
}

bool ReplayBuffer::compareStateWithAlternative( const State &state ) const
{
    const auto &alternativeSplit = _alternativeExperiences[_numExperiences - 1];
    auto alternativeState = alternativeSplit.get()->stateBeforeSplit;
    alternativeState.updateConstraintPhase( alternativeSplit.get()->plConstraint, 0 );
    return alternativeState.getData() == state.getData();
}