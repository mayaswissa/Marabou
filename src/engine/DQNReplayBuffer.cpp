#include "DQNReplayBuffer.h"

#include <Debug.h>
#include <memory>
#include <random>

ReplayBuffer::ReplayBuffer( const unsigned actionSize,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _actionSize( actionSize )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
{
}

void ReplayBuffer::pushActionEntry( const Action &action,
                                    const State &stateBeforeAction,
                                    const State &stateAfterAction,
                                    const unsigned depth,
                                    const unsigned numSplits )
{
    auto *actionEntry =
        new ActionsStack( action, stateBeforeAction, stateAfterAction, depth, numSplits );
    _actionsStack.append( actionEntry );
    printf( "replay buffer: add action entry, depth %u\n", _actionsStack.size() );
    fflush( stdout );
}

void ReplayBuffer::handleDone( const State &currentState,
                               const bool success,
                               const unsigned stackDepth,
                               const unsigned numSplits )
{
    // go over all actions in actionsStack and move them to revisitExperiences
    // no need to go over alternative actions since they did not occur.
    while ( !_actionsStack.empty() )
    {
        ActionsStack *actionEntry = _actionsStack.back();
        while ( !actionEntry->_activeActions.empty() )
        {
            pushToRevisit( currentState, stackDepth, numSplits, actionEntry );
        }
        delete _actionsStack.back();
        _actionsStack.popBack();
        printf( "replay buffer: pop action entry, depth %u\n", _actionsStack.size() );
        fflush( stdout );
    }
    _revisitExperiences.back().get()->_done = true;
    _revisitExperiences.back().get()->_reward = success ? 1 : -1;
}

void ReplayBuffer::pushToRevisit( const State &stateAfterAction,
                                  const unsigned depth,
                                  const unsigned numSplits,
                                  ActionsStack *actionEntry )
{
    auto activeAction = actionEntry->_activeActions.back();
    double reward = ( static_cast<double>( activeAction._splitsBeforeActiveAction ) -
                      static_cast<double>( numSplits ) ) /
                    activeAction._action.getNumPlConstraints();
    addToRevisitExperiences( activeAction._stateBeforeAction,
                             activeAction._action,
                             reward,
                             stateAfterAction,
                             false,
                             depth,
                             numSplits,
                             false );

    actionEntry->_activeActions.popBack();
}

// go to next alternative action available in actionsStack.
void ReplayBuffer::applyNextAction( const State &stateAfterAction,
                                    const unsigned depth,
                                    const unsigned numSplits,
                                    unsigned &numInconsistent )
{
    if ( _actionsStack.empty() )
    {
        handleDone( stateAfterAction, true, depth, numSplits );
        return;
    }

    ActionsStack *actionEntry;
    //  no alternative splits for previous actions - pop this entry and move activeActions to
    //  revisit Buffer.
    printf( "ReplayBuffer::applyNextAction\n" );
    fflush( stdout );
    while ( numInconsistent > 0 )
    {
        while ( _actionsStack.back()->_alternativeActions.empty() )
        {
            actionEntry = _actionsStack.back();
            // move activeSplit to revisit buffer.
            while ( !actionEntry->_activeActions.empty() )
            {
                pushToRevisit( stateAfterAction, depth, numSplits, actionEntry );
                printf( "replay buffer: applyNextAction, pop activeAction\n" );
                fflush( stdout );
            }
            delete _actionsStack.back();
            _actionsStack.popBack();
            printf( "replay buffer: pop entry, depth after pop: %u\n", _actionsStack.size() );
            fflush( stdout );

            if ( _actionsStack.empty() )
            {
                handleDone( stateAfterAction, true, depth, numSplits );
                return;
            }
        }

        // alternative action exists - push it to activeSplits with current numSplits:
        actionEntry = _actionsStack.back();
        auto action = actionEntry->_alternativeActions.begin();
        actionEntry->_activeActions.append( ActiveAction( *action,
                                                          actionEntry->_stateBeforeAction,
                                                          stateAfterAction,
                                                          actionEntry->_depthBeforeAction,
                                                          numSplits ) );
        actionEntry->_alternativeActions.erase( action );
        numInconsistent--;
        printf( "replay buffer: erased alternative, move it to active, depth: %u\n",
                _actionsStack.size() );
        fflush( stdout );
    }
}


void ReplayBuffer::addToRevisitExperiences( const State &state,
                                            const Action &action,
                                            double reward,
                                            const State &nextState,
                                            const bool done,
                                            unsigned depth,
                                            unsigned numSplits,
                                            bool changeReward )
{
    if ( _revisitExperiences.size() >= _bufferSize )
        _revisitExperiences.pop_front();


    auto experience = std::make_unique<Experience>(
        state, action, reward, nextState, done, depth, numSplits, changeReward );
    _revisitExperiences.push_back( std::move( experience ) );
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

    if ( _batchSize == 0 || _revisitExperiences.empty() )
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
    return _revisitExperiences.size();
}

unsigned ReplayBuffer::getBatchSize() const
{
    return _batchSize;
}

int ReplayBuffer::getActionStackSize() const
{
    if (_actionSize)
        return _actionsStack.size();
    return 0;
}

