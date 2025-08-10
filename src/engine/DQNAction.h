#ifndef DQNACTION_H
#define DQNACTION_H

#include "PiecewiseLinearConstraint.h"
#undef Warning
#include <torch/torch.h>
class Action
{
public:
    Action( unsigned numPhases, unsigned numPlConstraints );
    Action( unsigned numPhases, unsigned numPlConstraints, unsigned plConstraintActionIndex, unsigned phaseIndex );
    Action( const Action &other );
    Action &operator=( Action &&other ) noexcept;
    Action &operator=( const Action &other );
    unsigned getPlConstraintAction() const;
    torch::Tensor actionToTensor() const;
    unsigned getNumPhases() const;
    unsigned getNumPlConstraints() const;
    unsigned getActionPlConstraintIndex() const;
    unsigned getActionPhase() const;

private:
    unsigned _numPhases;
    unsigned _numPlConstraints;
    unsigned _plConstraintActionIndex;
    unsigned _phaseActionIndex;
};


#endif // DQNACTION_H
