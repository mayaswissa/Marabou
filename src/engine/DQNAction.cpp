#include "DQNActoin.h"

Action::Action( unsigned numPhases, unsigned numPlConstraints )
    : _numPhases( numPhases )
    , _numPlConstraints( numPlConstraints )
    , _plConstraintActionIndex( 0 )
    , _assignmentIndex( 0 )
{
}
Action::Action( unsigned numPhases,
                unsigned numPlConstraints,
                unsigned plConstraintActionIndex,
                unsigned assignmentIndex )
    : _numPhases( numPhases )
    , _numPlConstraints( numPlConstraints )
    , _plConstraintActionIndex( plConstraintActionIndex )
    , _assignmentIndex( assignmentIndex )
{
}
Action::Action( const Action &other )
    : _numPhases( other.getNumPhases() )
    , _numPlConstraints( other.getNumPlConstraints() )
    , _plConstraintActionIndex( other.getPlConstraintActionIndex() )
    , _assignmentIndex( other.getAssignmentIndex() )
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
unsigned Action::getPlConstraintActionIndex() const
{
    return _plConstraintActionIndex;
}

unsigned Action::getAssignmentIndex() const
{
    return _assignmentIndex;
}

Action &Action::operator=( Action &&other ) noexcept
{
    if ( this != &other )
    {
        _numPhases = other.getNumPhases();
        _plConstraintActionIndex = other.getPlConstraintActionIndex();
        _assignmentIndex = other.getAssignmentIndex();
    }
    return *this;
}

Action &Action::operator=( const Action &other )
{
    if ( this != &other )
    {
        _numPhases = other.getNumPhases();
        _plConstraintActionIndex = other.getPlConstraintActionIndex();
        _assignmentIndex = other.getAssignmentIndex();
    }
    return *this;
}

unsigned Action::getPlConstraintAction() const
{
    return _plConstraintActionIndex;
}

unsigned Action::getActionPhase() const
{
    return _assignmentIndex;
}
torch::Tensor Action::actionToTensor() const
{
    int combinedIndex = static_cast<int>( _plConstraintActionIndex ) * _numPhases +
                        static_cast<int>( _assignmentIndex );
    return torch::tensor( { combinedIndex }, torch::dtype( torch::kInt64 ) );
}
