#ifndef DQNSTATE_H
#define DQNSTATE_H

#include <vector>
#include <torch/torch.h>

class State {
public:
    State( unsigned numConstraints, unsigned numPhases );
    State(const State& other);
    State &operator=( const State &other );

    torch::Tensor toTensor() const;
    void updateConstraintPhase( unsigned constraintIndex, unsigned newPhase );
    void updateBounds( unsigned constraintIndex, double upperBound, double lowerBound );
    const std::vector<std::vector<double>>& getData() const;

private:
    // each inner vector represents a pl-constraint in one-hot encoding:
    // a single 1 indicating the current phase and 0s elsewhere.
    std::vector<std::vector<double>> _stateData;
    unsigned _numPhases;
};

#endif