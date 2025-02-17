#ifndef DQNSTATE_H
#define DQNSTATE_H

#include <vector>
#include <torch/torch.h>
enum DQNPhases : unsigned {
    DQN_RELU_NOT_FIXED = 0,
    DQN_RELU_ACTIVE = 1,
    DQN_RELU_INACTIVE = 2,
    DQN_RELU_OFF = 3,
    DQN_NUM_PHASES
};

class State {
public:
    State( unsigned numConstraints );
    State(const State& other);
    State &operator=( const State &other );

    torch::Tensor toTensor() const;
    void updateConstraintPhase( unsigned constraintIndex, unsigned newPhase );
    void updateBounds( unsigned constraintIndex, double upperBound, double lowerBound );
    const std::vector<std::vector<double>> &getData() const;
    bool constraintActive( unsigned constraintIndex ) const;

private:
    // each inner vector represents a pl-constraint in one-hot encoding:
    // a single 1 indicating the current phase and 0s elsewhere.
    std::vector<std::vector<double>> _stateData;
    unsigned _numPhases;
};

#endif