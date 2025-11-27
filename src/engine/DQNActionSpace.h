#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H
#include <utility>
#include "Vector.h"
class ActionSpace {
public:
    ActionSpace(unsigned numConstraints, unsigned numPhases);

    unsigned getActionIndex(unsigned constraintIndex, unsigned phaseIndex ) const;

    std::pair<unsigned, unsigned> decodeActionIndex(unsigned actionIndex) const;
    unsigned getNumActions() const;
    unsigned getNumPhases() const;
    unsigned getNumConstraints() const;

private:
    unsigned int _numConstraints;
    unsigned int _numPhases;
    Vector<int> _actionIndices;
};

#endif //ACTIONSPACE_H
