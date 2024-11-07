#include "DQNActionSpace.h"

#include <utility>

ActionSpace::ActionSpace(unsigned numConstraints,unsigned numPhases)
: numConstraints(numConstraints), numPhases(numPhases) {
    for (int i = 0; i < numConstraints; ++i) {
        for (int j = 0; j < numPhases; ++j) {
            actionIndices.append(i * numPhases + j);
        }
    }
}
unsigned ActionSpace::getActionIndex(unsigned constraintIndex, unsigned phaseIndex ) const
{
    return constraintIndex * numPhases + phaseIndex;
}
std::pair<unsigned, unsigned> ActionSpace::decodeActionIndex(unsigned actionIndex) const {
    unsigned constraintIndex = actionIndex / numPhases;
    unsigned phaseIndex = actionIndex % numPhases;
    return {constraintIndex, phaseIndex};
}

