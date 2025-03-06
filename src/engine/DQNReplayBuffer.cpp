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
    _actions = torch::zeros( { static_cast<long>( bufferSize ), 1 }, torch::kFloat32 );
    _states = torch::zeros( { static_cast<long>( bufferSize ), _numConstraints, NUM_FEATURES },
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
                                    const unsigned numSplitsBeforeAction,
                                    const double soiScoreBeforeAction )
{
    auto *actionEntry = new ActionEntry( action, stateBeforeAction, numSplitsBeforeAction, soiScoreBeforeAction );
    _actionsStack.append( actionEntry );
}

void ReplayBuffer::handleDone( const State &currentState,
                               const unsigned numSplits,
                               const double soiScore )
{
    // Go over all actions in actionsStack and move them to revisitExperiences
    while ( !_actionsStack.empty() )
    {
        ActionEntry *actionEntry = _actionsStack.back();
        // no need to insert alternative actions.
        while ( !actionEntry->_activeActions.empty() )
            moveActionToRevisitBuffer( currentState, numSplits, actionEntry, soiScore );

        delete _actionsStack.back();
        _actionsStack.popBack();
    }
}

void ReplayBuffer::moveActionToRevisitBuffer( const State &stateAfterAction,
                                              const unsigned numSplitsAfterAction,
                                              ActionEntry *actionEntry,
                                              double soiScoreAfterAction )
{
    const auto activeAction = actionEntry->_activeActions.back();
    double splitsReward = ( static_cast<double>( activeAction._splitsBeforeActiveAction ) -
                            static_cast<double>( numSplitsAfterAction ) ) /
                          activeAction._action.getNumPlConstraints();
    if (splitsReward == 0)
    {
        actionEntry->_activeActions.popBack();
        return;
    }
    double soiReward = -2; // this is the reward for soiBefore < soiAfter
    if (activeAction._soiScoreBeforeActiveAction > soiScoreAfterAction)
    {
        soiReward = soiScoreAfterAction - activeAction._soiScoreBeforeActiveAction;
    }
    else if (activeAction._soiScoreBeforeActiveAction == soiScoreAfterAction)
        soiReward = -0.5;
    soiReward =
        std::copysign( std::log( 1.0 + std::abs( soiReward ) / 10.0 + 1e-8 ), soiReward );
    // std::cout << "soi reward : " << soiReward << std::endl;
    const auto reward =   soiReward;

    addExperienceToRevisitBuffer(
        activeAction._stateBeforeAction, activeAction._action, reward, stateAfterAction, false );
    actionEntry->_activeActions.popBack();
}

// go to next alternative action available in actionsStack.
void ReplayBuffer::applyNextAction( const State &stateAfterAction,
                                    const unsigned numSplits,
                                    unsigned &numInconsistent,
                                    const double soiScore )
{
    if ( _actionsStack.empty() )
        return;

    ActionEntry *actionEntry;

    while ( numInconsistent > 0 )
    {
        //  no alternative splits for this action - pop the entry and move activeActions to
        //  revisitExperiences buffer.
        while ( _actionsStack.back()->_alternativeActions.empty() )
        {
            actionEntry = _actionsStack.back();
            while ( !actionEntry->_activeActions.empty() )
            {
                moveActionToRevisitBuffer( stateAfterAction, numSplits, actionEntry, soiScore );
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
            ActiveAction( *action, actionEntry->_stateBeforeAction, numSplits, soiScore ) ); // todo check what value the
                                                                    // state before should have
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
    sampledIndices.insert(
        sampledIndices.end(), indices.begin(), indices.begin() + currentBatchSize );
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
