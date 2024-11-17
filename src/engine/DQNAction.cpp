#include "DQNActoin.h"

Action::Action( unsigned plConstraintActionIndex, unsigned assignmentIndex )
    : _plConstraintActionIndex( plConstraintActionIndex )
    , _assignmentIndex( assignmentIndex )
{}

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
    return torch::tensor( { static_cast<int>( _plConstraintActionIndex ),
                            static_cast<int>( _assignmentIndex ) },
                          torch::dtype( torch::kInt64 ) );
}


