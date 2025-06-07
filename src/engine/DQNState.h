#ifndef DQNSTATE_H
#define DQNSTATE_H

#include <vector>
#include <torch/torch.h>
enum DQNPhases : unsigned {
    DQN_RELU_NOT_FIXED = 0,
    DQN_RELU_ACTIVE = 1,
    DQN_RELU_INACTIVE = 2,
    DQN_NUM_PHASES
};

enum DQNFeatures : unsigned {
    DQN_RELU_NOT_FIXED_VALUE = 0,
    DQN_RELU_ACTIVE_VALUE = 1,
    DQN_RELU_INACTIVE_VALUE = 2,
    SOI_ACTIVE_SCORE = 3,
    SOI_INACTIVE_SCORE = 4,
    POLARITY_SCORE = 5,
    BaBsr_SCORE = 6,

    NUM_FEATURES
};

class State {
public:
    State( unsigned numConstraints );
    State(const State& other);
    State &operator=( const State &other );
    void debug( int &sum ) const;

    torch::Tensor toTensor() const;
    void updateConstraintPhase( unsigned constraintIndex, unsigned newPhase );
    void updateSoIScoreForAgent( unsigned constraintIndex, double SoiActiveScore, double SoiInactiveScore );
    void updateBounds( unsigned constraintIndex, double upperBound, double lowerBound );
    void updatePolarity( unsigned constraintIndex, double polarityScore );
    const std::vector<std::vector<double>> &getData() const;
    unsigned getNumConstraints() const;
    // Accessor
    const std::vector<double>& getRawData() const { return _stateData; }
private:
    // each inner vector represents a pl-constraint in one-hot encoding:
    // a single 1 indicating the current phase and 0s elsewhere.
    std::vector<double> _stateData;   // length = numConstraints * NUM_FEATURES
    unsigned _numConstraints;
    unsigned _numPhases;
};

#endif