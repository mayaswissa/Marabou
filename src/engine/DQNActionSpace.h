#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H
#include <utility>
#include "Vector.h"
class ActionSpace {
public:
    ActionSpace(unsigned numConstraints, unsigned numPhases);

    // Get the encoded action index
    unsigned getActionIndex(unsigned constraintIndex, unsigned phaseIndex ) const;

    // Example method to decode an action index back to its parts
    std::pair<unsigned, unsigned> decodeActionIndex(unsigned actionIndex) const;

private:
    unsigned int numConstraints;
    unsigned int numPhases;
    Vector<int> actionIndices;
};

#endif //ACTIONSPACE_H
