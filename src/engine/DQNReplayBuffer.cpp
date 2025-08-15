#include "DQNReplayBuffer.h"

#include "RandomGlobals.h"

#include <random>

ReplayBuffer::ReplayBuffer( const unsigned numConstraints,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _numConstraints( numConstraints )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
    , _fakeActionIndex( numConstraints + 1 )
    , _size( 0 )
    , _writePosition( 0 )
    , _priorities( bufferSize, 0.0f )
    , _sumTree( bufferSize * 2, 0.0f )
    , _isDemo( bufferSize, false )
    , _maxPriority( 1.0f )
    , _demoFactor( 2.0f )
    , _eps( 1e-6f )
    , _beta( 0.4f )
    , _betaInc( ( 1.0f - 0.4f ) / 100000.0f )
{
    _actions =
        torch::zeros( { static_cast<long>( bufferSize ), 1 }, torch::dtype( torch::kInt64 ) );
    _states = torch::zeros( { static_cast<long>( bufferSize ),
                              static_cast<long>( _numConstraints ),
                              static_cast<long>( TOTAL_FEATURES ) },
                            torch::kFloat32 );
    _rewards = torch::zeros( { static_cast<long>( bufferSize ) }, torch::kFloat32 );
    _nextStates = torch::zeros( { static_cast<long>( bufferSize ),
                                  static_cast<long>( _numConstraints ),
                                  static_cast<long>( TOTAL_FEATURES ) },
                                torch::kFloat32 );
    _dones = torch::zeros( { static_cast<long>( bufferSize ) }, torch::kInt );
}

void ReplayBuffer::pushFakeActionEntry( const State &stateBeforeAction,
                                        const unsigned numSplitsBeforeAction )
{
    const auto fakeAction = std::make_unique<Action>(
        DQN_NUM_PHASES, _numConstraints, _fakeActionIndex, RELU_PHASE_ACTIVE );
    auto *actionEntry = new ActionEntry(
        *fakeAction, stateBeforeAction, numSplitsBeforeAction, true, false, false );
    _actionsStack.append( actionEntry );
}

void ReplayBuffer::pushActionEntry( const Action &action,
                                    const State &stateBeforeAction,
                                    const unsigned numSplitsBeforeAction,
                                    const bool demo,
                                    const bool done )
{
    auto *actionEntry =
        new ActionEntry( action, stateBeforeAction, numSplitsBeforeAction, false, demo, done );
    _actionsStack.append( actionEntry );
}

void ReplayBuffer::handleDone( const State &currentState, const unsigned numSplits )
{
    // Go over all actions in actionsStack and move them to revisitExperiences
    while ( !_actionsStack.empty() )
    {
        ActionEntry *actionEntry = _actionsStack.back();
        // no need to insert alternative actions.
        while ( !actionEntry->_activeActions.empty() ) // todo handle done iters differently
            moveActionToRevisitBuffer( currentState, numSplits, actionEntry, actionEntry->_done );

        delete _actionsStack.back();
        _actionsStack.popBack();
    }
}

double ReplayBuffer::potentialSubtreeSize() const
{
    const unsigned currentDepth = getActionStackSize();
    if ( currentDepth >= _numConstraints )
        return 0.0;
    return ( _numConstraints - currentDepth ) * std::log( 2.0L );
}


void ReplayBuffer::moveActionToRevisitBuffer( const State &stateAfterAction,
                                              const unsigned numSplitsAfterAction,
                                              ActionEntry *actionEntry,
                                              const bool done )
{
    const auto activeAction = actionEntry->_activeActions.back();
    if ( actionEntry->_isFake )
    {
        actionEntry->_activeActions.popBack();
        return;
    }
    const double deltaSplit = static_cast<double>( activeAction._splitsBeforeActiveAction ) -
                              static_cast<double>( numSplitsAfterAction );

    if ( deltaSplit == 0 && !done )
    {
        actionEntry->_activeActions.popBack();
        return;
    }

    auto reward = potentialSubtreeSize() != 0 ? deltaSplit / potentialSubtreeSize() : 0;
    double alpha = 10.0;
    reward = std::copysign( std::tanh( alpha * std::abs( reward ) ), reward );
    addExperienceToRevisitBuffer( activeAction._stateBeforeAction,
                                  activeAction._action,
                                  reward,
                                  stateAfterAction,
                                  done,
                                  actionEntry->_isDemo );
    actionEntry->_activeActions.popBack();
}

// go to next alternative action available in actionsStack.
void ReplayBuffer::applyNextAction( const State &stateAfterAction,
                                    const unsigned numSplits,
                                    unsigned &numInconsistent )
{
    if ( _actionsStack.empty() )
        return;

    while ( numInconsistent > 0 )
    {
        //  no alternative splits for this action - pop the entry and move activeActions to
        //  revisitExperiences buffer.
        while ( !_actionsStack.empty() && _actionsStack.back()->_alternativeActions.empty() )
        {
            auto *entry = _actionsStack.back();
            // Finalize exactly the subtree that just finished at this entry.
            if ( !entry->_activeActions.empty() )
                moveActionToRevisitBuffer( stateAfterAction, numSplits, entry );

            // Entry has no opposite left; remove it and go up.
            delete entry;
            _actionsStack.popBack();
        }
        if ( _actionsStack.empty() )
            return;

        auto *actionEntry = _actionsStack.back();
        // Finalize the subtree that just finished.
        ASSERT( !actionEntry->_activeActions.empty() );
        const auto &finishedActiveAction  = actionEntry->_activeActions.back();
        const State   preState  = finishedActiveAction._stateBeforeAction;
        const unsigned preSplits = finishedActiveAction._splitsBeforeActiveAction;

        const auto before = actionEntry->_activeActions.size();
        moveActionToRevisitBuffer(stateAfterAction, numSplits, actionEntry);
        ASSERT(actionEntry->_activeActions.size() + 1 == before);


        // Schedule the alternative action from the same branching.
        if ( !actionEntry->_alternativeActions.empty() )
        {
            const auto it = actionEntry->_alternativeActions.begin();
            actionEntry->_activeActions.append( ActiveAction( *it, preState, preSplits ) );
            actionEntry->_alternativeActions.erase( it );
            --numInconsistent;
            return;
        }
    }
}

void ReplayBuffer::addExperienceToRevisitBuffer( const State &state,
                                                 const Action &action,
                                                 double reward,
                                                 const State &nextState,
                                                 const bool done,
                                                 bool isDemo )
{
    const auto stateTensor = state.toTensor();
    const auto actionTensor = action.actionToTensor();
    const auto nextStateTensor = nextState.toTensor();
    _states.index_put_(
        { static_cast<long>( _writePosition ), torch::indexing::Slice(), torch::indexing::Slice() },
        stateTensor );
    _actions.index_put_( { static_cast<long>( _writePosition ), 0 }, actionTensor );
    _rewards.index_put_( { static_cast<long>( _writePosition ) }, static_cast<double>( reward ) );
    _nextStates.index_put_(
        { static_cast<long>( _writePosition ), torch::indexing::Slice(), torch::indexing::Slice() },
        nextStateTensor );
    _dones.index_put_( { static_cast<long>( _writePosition ) }, done ? 1 : 0 );

    _isDemo[_writePosition] = isDemo;
    float p = isDemo ? _epsDemo : _epsAgent;

    updatePriority( _writePosition, p );
    _maxPriority = std::max( _maxPriority, p );
    _writePosition = ( _writePosition + 1 ) % _bufferSize;
    if ( _size < _bufferSize )
        ++_size;
}

SampledBatch ReplayBuffer::sample()
{
    SampledBatch batch;
    if ( _size < _batchSize )
        return batch;

    float total = _sumTree[1];
    float segment = total / _batchSize;
    for ( unsigned i = 0; i < _batchSize; ++i )
    {
        float a = segment * i;
        float b = segment * ( i + 1 );
        float s = RandomGlobals::instance().rand01() * ( b - a ) + a;
        unsigned ti = 1;
        while ( ti < _bufferSize )
        {
            if ( s <= _sumTree[2 * ti] )
                ti *= 2;
            else
            {
                s -= _sumTree[2 * ti];
                ti = 2 * ti + 1;
            }
        }
        unsigned idx = ti - _bufferSize;
        float Pj = _priorities[idx] / total;
        float w = std::pow( _size * Pj + _eps, -_beta );

        batch.indices.push_back( idx );
        batch.weights.push_back( w );
        batch.isDemo.push_back( _isDemo[idx] );
    }
    // anneal beta
    _beta = std::min( 1.0f, _beta + _betaInc );
    return batch;
}

void ReplayBuffer::updatePriority( unsigned idx, float newP )
{
    _priorities[idx] = newP;
    unsigned ti = idx + _bufferSize;
    _sumTree[ti] = newP;
    rebuildTree( ti );
    _maxPriority = std::max( _maxPriority, newP );
}

void ReplayBuffer::rebuildTree( unsigned ti )
{
    while ( ti > 1 )
    {
        ti /= 2;
        _sumTree[ti] = _sumTree[2 * ti] + _sumTree[2 * ti + 1];
    }
}
unsigned ReplayBuffer::getNumRevisitExperiences() const
{
    return _size;
}

unsigned ReplayBuffer::getBatchSize() const
{
    return _batchSize;
}

int ReplayBuffer::getActionStackSize() const
{
    return _actionsStack.size();
}

torch::Tensor ReplayBuffer::getStates()
{
    return _states;
}
torch::Tensor ReplayBuffer::getNextStates()
{
    return _nextStates;
}
torch::Tensor ReplayBuffer::getActions()
{
    return _actions;
}
torch::Tensor ReplayBuffer::getRewards()
{
    return _rewards;
}
torch::Tensor ReplayBuffer::getDones()
{
    return _dones;
}

float ReplayBuffer::getDemoFactor() const
{
    return _demoFactor;
}

float ReplayBuffer::getEpsilonDemo() const
{
    return _epsDemo;
}

float ReplayBuffer::getEpsilonAgent() const
{
    return _epsAgent;
}