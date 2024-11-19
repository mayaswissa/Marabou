#include "DQNActoin.h"

Action::Action( unsigned numPhases, unsigned plConstraintActionIndex, unsigned assignmentIndex )
    : _numPhases( numPhases )
    , _plConstraintActionIndex( plConstraintActionIndex )
    , _assignmentIndex( assignmentIndex )
{
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
