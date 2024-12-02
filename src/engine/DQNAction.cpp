#include "DQNActoin.h"

Action::Action(unsigned numPhases) : _numPhases( numPhases ), _plConstraintActionIndex( 0 ) , _assignmentIndex( 0 )
{
}
Action::Action( unsigned numPhases, unsigned plConstraintActionIndex, unsigned assignmentIndex )
    : _numPhases( numPhases )
    , _plConstraintActionIndex( plConstraintActionIndex )
    , _assignmentIndex( assignmentIndex )
{
}
Action::Action(const Action& other)
        : _numPhases(other.getNumPhases()),
          _plConstraintActionIndex(other.getPlConstraintActionIndex()),
          _assignmentIndex(other.getAssignmentIndex()) {}



unsigned Action::getNumPhases() const
{
    return _numPhases;
}

unsigned Action::getPlConstraintActionIndex() const
{
    return _plConstraintActionIndex;
}

unsigned Action::getAssignmentIndex() const
{
    return _assignmentIndex;
}

Action& Action::operator=(Action&& other) noexcept {
    if (this != &other)
    {
        _numPhases = other.getNumPhases();
        _plConstraintActionIndex = other.getPlConstraintActionIndex();
        _assignmentIndex = other.getAssignmentIndex();
    }
    return *this;
}

Action& Action::operator=(const Action& other) {
    if (this != &other)
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

unsigned Action::getAssignmentStatus() const
{
    return _assignmentIndex;
}
torch::Tensor Action::actionToTensor() const
{
    int combinedIndex = static_cast<int>(_plConstraintActionIndex) * _numPhases +
                        static_cast<int>(_assignmentIndex);
    return torch::tensor({combinedIndex}, torch::dtype(torch::kInt64));
}
