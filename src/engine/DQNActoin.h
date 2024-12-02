#ifndef DQNACTOIN_H
#define DQNACTOIN_H

#include "PiecewiseLinearConstraint.h"
#undef Warning
#include <torch/torch.h>
class Action
{
public:
    Action( unsigned numPhases );
    Action( unsigned numPhases, unsigned plConstraintActionIndex, unsigned assignmentIndex );
    Action( const Action &other );
    Action &operator=( Action &&other ) noexcept;
    Action &operator=( const Action &other );
    unsigned getPlConstraintAction() const;

    unsigned getAssignmentStatus() const;

    torch::Tensor actionToTensor() const;

    unsigned getNumPhases() const;
    unsigned getPlConstraintActionIndex() const;
    unsigned getAssignmentIndex() const;

private:
    unsigned _numPhases;
    unsigned _plConstraintActionIndex;
    unsigned _assignmentIndex;
};


#endif // DQNACTOIN_H
