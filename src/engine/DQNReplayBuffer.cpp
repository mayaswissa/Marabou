#include "DQNReplayBuffer.h"
#include <random>

ReplayBuffer::ReplayBuffer( const unsigned numConstraints,
                            const unsigned bufferSize,
                            const unsigned batchSize )
    : _numConstraints( numConstraints )
    , _bufferSize( bufferSize )
    , _batchSize( batchSize )
    , _size( 0 )
    , _writePosition( 0 )
{
    _actions = torch::zeros( { static_cast<long>( bufferSize) , 1 }, torch::kFloat32 );
    _states = torch::zeros( { static_cast<long>( bufferSize ),
                               _numConstraints ,
                              NUM_FEATURES  },
                            torch::kFloat32 );
    _rewards = torch::zeros( { static_cast<long>( bufferSize ) }, torch::kFloat32 );
    _nextStates = torch::zeros( { static_cast<long>( bufferSize ),
                                  static_cast<long>( _numConstraints ),
                                  static_cast<long>( NUM_FEATURES ) },
                                torch::kFloat32 );
    _dones = torch::zeros( { static_cast<long>( bufferSize ) }, torch::kInt );
}

void ReplayBuffer::pushActionEntry( const Action &action,
                                    const State &stateBeforeAction,
                                    const State &stateAfterAction,
                                    const unsigned numSplits )
{
    auto *actionEntry = new ActionsStack( action, stateBeforeAction, stateAfterAction, numSplits );
    _actionsStack.append( actionEntry );
}

void ReplayBuffer::handleDone( const State &currentState,
                               const unsigned numSplits,
                               const double prunedSubtrees )
{
    // Go over all actions in actionsStack and move them to revisitExperiences
    while ( !_actionsStack.empty() )
    {
        ActionsStack *actionEntry = _actionsStack.back();
        // no need to insert alternative actions.
        while ( !actionEntry->_activeActions.empty() )
            moveActionToRevisitBuffer( currentState, numSplits, actionEntry, prunedSubtrees );

        delete _actionsStack.back();
        _actionsStack.popBack();
    }
}

void ReplayBuffer::moveActionToRevisitBuffer( const State &stateAfterAction,
                                              const unsigned numSplits,
                                              ActionsStack *actionEntry,
                                              const double prunedSubtrees )
{
    const auto activeAction = actionEntry->_activeActions.back();
    double splitsReward = ( static_cast<double>( activeAction._splitsBeforeActiveAction ) -
                            static_cast<double>( numSplits ) ) /
                          activeAction._action.getNumPlConstraints();
    splitsReward =
        std::copysign( std::log( 1.0 + std::abs( splitsReward ) / 10.0 + 1e-8 ), splitsReward );
    const auto reward = GlobalConfiguration::DQN_ALPHA_REWARDS * splitsReward +
                        ( 1.0 - GlobalConfiguration::DQN_ALPHA_REWARDS ) * prunedSubtrees;

    addExperienceToRevisitBuffer(
        activeAction._stateBeforeAction, activeAction._action, reward, stateAfterAction, false );
    actionEntry->_activeActions.popBack();
}

// go to next alternative action available in actionsStack.
void ReplayBuffer::applyNextAction( const State &stateAfterAction,
                                    const unsigned numSplits,
                                    unsigned &numInconsistent,
                                    const double prunedSubtrees )
{
    if ( _actionsStack.empty() )
        return;

    ActionsStack *actionEntry;

    while ( numInconsistent > 0 )
    {
        //  no alternative splits for this action - pop the entry and move activeActions to
        //  revisitExperiences buffer.
        while ( _actionsStack.back()->_alternativeActions.empty() )
        {
            actionEntry = _actionsStack.back();
            while ( !actionEntry->_activeActions.empty() )
            {
                moveActionToRevisitBuffer(
                    stateAfterAction, numSplits, actionEntry, prunedSubtrees );
            }
            delete _actionsStack.back();
            _actionsStack.popBack();

            if ( _actionsStack.empty() )
                return;
        }

        // alternative action exists - push it to activeSplits with current numSplits:
        actionEntry = _actionsStack.back();
        auto action = actionEntry->_alternativeActions.begin();
        actionEntry->_activeActions.append(
            ActiveAction( *action, actionEntry->_stateBeforeAction, stateAfterAction, numSplits ) );
        actionEntry->_alternativeActions.erase( action );
        numInconsistent--;
    }
}

void ReplayBuffer::addExperienceToRevisitBuffer( const State &state,
                                                 const Action &action,
                                                 double reward,
                                                 const State &nextState,
                                                 const bool done )
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

    _writePosition = ( _writePosition + 1 ) % _bufferSize;
    if ( _size < _bufferSize )
        ++_size;
}

std::vector<unsigned> ReplayBuffer::sample() const
{
    std::vector<unsigned> sampledIndices;

    if ( _batchSize == 0 || _size < _batchSize * GlobalConfiguration::DQN_MIN_SAMPLE_SIZE )
    {
        return sampledIndices;
    }
    const unsigned startIndex = 0;
    const unsigned endIndex = _size - 1;

    const unsigned rangeSize = endIndex - startIndex + 1;
    const unsigned currentBatchSize = std::min( _batchSize, rangeSize );

    Vector<unsigned> indices( rangeSize );
    std::iota( indices.begin(), indices.end(), startIndex );

    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( indices.begin(), indices.end(), g );
    sampledIndices.insert(sampledIndices.end(), indices.begin(), indices.begin() + currentBatchSize);
    return sampledIndices;
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
    if ( !_actionsStack.empty() )
        return _actionsStack.size();
    return 0;
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
