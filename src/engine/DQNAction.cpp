#include "DQNAction.h"

Action::Action( const unsigned numPhases, const unsigned numPlConstraints )
    : _numPhases( numPhases )
    , _numPlConstraints( numPlConstraints )
    , _plConstraintActionIndex( 0 )
    , _phaseActionIndex( 0 )
{
}
Action::Action( const unsigned numPhases,
                const unsigned numPlConstraints,
                const unsigned plConstraintActionIndex,
                const unsigned phaseIndex )
    : _numPhases( numPhases )
    , _numPlConstraints( numPlConstraints )
    , _plConstraintActionIndex( plConstraintActionIndex )
    , _phaseActionIndex( phaseIndex )
{
}
Action::Action( const Action &other )
    : _numPhases( other.getNumPhases() )
    , _numPlConstraints( other.getNumPlConstraints() )
    , _plConstraintActionIndex( other.getActionPlConstraintIndex() )
    , _phaseActionIndex( other.getActionPhase() )
{
}


unsigned Action::getNumPhases() const
{
    return _numPhases;
}

unsigned Action::getNumPlConstraints() const
{
    return _numPlConstraints;
}
unsigned Action::getActionPlConstraintIndex() const
{
    return _plConstraintActionIndex;
}

Action &Action::operator=( Action &&other ) noexcept
{
    if ( this != &other )
    {
        _numPhases = other.getNumPhases();
        _plConstraintActionIndex = other.getActionPlConstraintIndex();
        _phaseActionIndex = other.getActionPhase();
        _numPlConstraints = other.getNumPlConstraints();
    }
    return *this;
}

Action &Action::operator=( const Action &other )
{
    if ( this != &other )
    {
        _numPhases = other.getNumPhases();
        _plConstraintActionIndex = other.getActionPlConstraintIndex();
        _phaseActionIndex = other.getActionPhase();
        _numPlConstraints = other.getNumPlConstraints();
    }
    return *this;
}

unsigned Action::getPlConstraintAction() const
{
    return _plConstraintActionIndex;
}

unsigned Action::getActionPhase() const
{
    return _phaseActionIndex;
}

torch::Tensor Action::actionToTensor() const
{
    int combinedIndex = static_cast<int>( _plConstraintActionIndex ) * _numPhases +
                        static_cast<int>( _phaseActionIndex );
    return torch::tensor( { combinedIndex }, torch::dtype( torch::kInt64 ) );
}
