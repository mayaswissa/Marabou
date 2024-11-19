#ifndef DQNACTOIN_H
#define DQNACTOIN_H

#include "PiecewiseLinearConstraint.h"
#undef Warning
#include <torch/torch.h>
class Action
{
public:
    Action( unsigned numPhases, unsigned plConstraintActionIndex, unsigned assignmentIndex );

    unsigned getPlConstraintAction() const;

    unsigned getAssignmentStatus() const;

    torch::Tensor actionToTensor() const;

private:
    const unsigned _numPhases;
    const unsigned _plConstraintActionIndex;
    const unsigned _assignmentIndex;
};


#endif // DQNACTOIN_H
