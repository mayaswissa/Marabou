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
    , _numRevisitedExperiences( 0 )
{
}

void ReplayBuffer::pushActionEntry( Action action,
                                    State stateBeforeAction,
                                    State stateAfterAction,
                                    unsigned depth,
                                    unsigned numSplits )
{
    ActionsStack *actionEntry = new ActionsStack( std::move( action ),
                                                  std::move( stateBeforeAction ),
                                                  std::move( stateAfterAction ),
                                                  depth,
                                                  numSplits );
    _actionsStack.append( actionEntry );
}

void ReplayBuffer::handleDone( State currentState, bool success, unsigned stackDepth, unsigned numSplits )
{
    // go over all actions in actionsStack and move them to revisitExperiences
    // no need to go over alternative actions since they did not occur.
    while (!_actionsStack.empty())
    {
        ActionsStack *actionEntry = _actionsStack.back();
        while (! actionEntry->_activeActions.empty())
        {
            auto activeAction = actionEntry->_activeActions.back();
            double reward = 0;
            auto progress = numSplits - activeAction._splitsBeforeActiveAction;
            if ( progress > 0 )
                reward = 1.0 / static_cast<double>( progress );
            auto experience =
                std::make_unique<Experience>( std::move( activeAction._stateBeforeAction ),
                                              std::move( activeAction._action ),
                                              reward,
                                              currentState,
                                              false,
                                              stackDepth,
                                              numSplits,
                                              false );
            _revisitExperiences.push_back( std::move( experience ) );
            actionEntry->_activeActions.popBack();
        }
        delete _actionsStack.back();
        _actionsStack.popBack();
    }

    _revisitExperiences.back().get()->_done = true;
    _revisitExperiences.back().get()->_reward = success ? 10 : -10;
}

// go to next alternative action available in actionsStack.
void ReplayBuffer::applyNextAction( State stateAfterAction, unsigned depth, unsigned numSplits )
{
    if ( _actionsStack.empty() )
    {
        handleDone(std::move(stateAfterAction), true, depth, numSplits); // todo check success - if no actions in actions stack - seems like unsat successfully
        return;
    }
    ActionsStack *actionEntry = _actionsStack.back();
    //  no alternative splits for previous actions - pop this entry and move activeActions to
    //  revisit Buffer.
    while ( actionEntry->_alternativeActions.empty() )
    {
        // move activeSplit to revisit buffer.
        while ( !actionEntry->_activeActions.empty() )
        {
            auto activeAction = actionEntry->_activeActions.back();
            double reward = 0;
            auto progress = numSplits - activeAction._splitsBeforeActiveAction;
            if ( progress > 0 )
                reward = 1.0 / static_cast<double>( progress );
            auto experience =
                std::make_unique<Experience>( std::move( activeAction._stateBeforeAction ),
                                              std::move( activeAction._action ),
                                              reward,
                                              stateAfterAction ,
                                              false,
                                              depth,
                                              numSplits,
                                              false );
            _revisitExperiences.push_back( std::move( experience ) );
            actionEntry->_activeActions.popBack();
        }
        delete _actionsStack.back();
        _actionsStack.popBack();

        if ( _actionsStack.empty() )
        {
            handleDone(stateAfterAction, true, depth, numSplits);
            return;
        }
        actionEntry = _actionsStack.back();
    }

        // alternative action exists - push it to activeSplits with current numSplits:
        actionEntry = _actionsStack.back();
        auto action = actionEntry->_alternativeActions.begin();
        actionEntry->_activeActions.append( ActiveAction( std::move( *action ),
                                                          actionEntry->_stateBeforeAction,
                                                          std::move( stateAfterAction ),
                                                          depth,
                                                          numSplits ) );
        actionEntry->_alternativeActions.erase( action );

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


Experience &ReplayBuffer::getRevisitExperienceAt( const unsigned index ) const
{
    if ( index >= getNumRevisitExperiences() )
        throw std::out_of_range( "Index out of range in revisited experiences" ); // todo error
    return *_revisitExperiences[index];
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

unsigned ReplayBuffer::getNumRevisitExperiences() const
{
    return _numRevisitedExperiences;
}

unsigned ReplayBuffer::getBatchSize() const
{
    return _batchSize;
}


