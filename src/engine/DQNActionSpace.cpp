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

unsigned ActionSpace::getActionIndex( const unsigned constraintIndex,
                                      const unsigned phaseIndex ) const
{
    return constraintIndex * _numPhases + phaseIndex;
}

std::pair<unsigned, unsigned> ActionSpace::decodeActionIndex( const unsigned actionIndex) const {
    unsigned constraintIndex = actionIndex / _numPhases;
    unsigned phaseIndex = actionIndex % _numPhases;
    return {constraintIndex, phaseIndex};
}

unsigned ActionSpace::getNumActions() const
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
