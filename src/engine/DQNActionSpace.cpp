#include "DQNActionSpace.h"

#include <utility>
#include <iostream>
ActionSpace::ActionSpace(unsigned numConstraints,unsigned numPhases)
: _numConstraints(numConstraints), _numPhases(numPhases) {
    for (unsigned i = 0; i < numConstraints; ++i) {
        for (unsigned j = 0; j < numPhases; ++j) {
            _actionIndices.append(i * numPhases + j);
        }
    }
}
unsigned ActionSpace::getActionIndex(unsigned constraintIndex, unsigned phaseIndex ) const
{
    return constraintIndex * _numPhases + phaseIndex;
}
std::pair<unsigned, unsigned> ActionSpace::decodeActionIndex(unsigned actionIndex) const {
    unsigned constraintIndex = actionIndex / _numPhases;
    unsigned phaseIndex = actionIndex % _numPhases;
    return {constraintIndex, phaseIndex};
}

unsigned ActionSpace::getSpaceSize() const
{
    return _numConstraints * _numPhases;
}
unsigned ActionSpace::getNumPhases() const
{
    return _numPhases;
}
unsigned ActionSpace::getNumConstraints() const
{
    return _numConstraints;
}
