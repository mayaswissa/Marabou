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
    DQN_RELU_LOWER_BOUND = 3,
    DQN_RELU_UPPER_BOUND = 4,
    SOI_ACTIVE_SCORE = 5,
    SOI_INACTIVE_SCORE = 6,
    POLARITY_SCORE = 7,
    BaBsr_SCORE = 8,

    NUM_LOCAL_FEATURES
};

enum GlobalFeatures : unsigned {
    GF_UNSTABLE_COUNT = 0,
    GF_TREE_DEPTH = 1,
    GF_SPLITS_SO_FAR = 2,

    NUM_GLOBAL_FEATURES
};
static constexpr unsigned TOTAL_FEATURES = NUM_LOCAL_FEATURES + NUM_GLOBAL_FEATURES;

class State
{
public:
    State( unsigned numConstraints );
    State( const State &other );
    State &operator=( const State &other );
    torch::Tensor toTensor() const;
    void updateConstraintPhase( unsigned constraintIndex, unsigned newPhase );
    void updateSoIScoreForAgent( unsigned constraintIndex,
                                 double SoiActiveScore,
                                 double SoiInactiveScore );
    void updateBounds( unsigned constraintIndex, double upperBound, double lowerBound );
    void updatePolarity( unsigned constraintIndex, double polarityScore );
    void updateBaBsrScore( unsigned constraintIndex, double BaBsrScore );
    const std::vector<float> &getRawData() const
    {
        return _stateData;
    }
    void updateGlobalFeatures( unsigned unstableCount, unsigned treeDepth, unsigned splitsSoFar );

private:
    // each inner vector represents a pl-constraint in one-hot encoding:
    // a single 1 indicating the current phase and 0s elsewhere.
    std::vector<float> _stateData;
    unsigned _numConstraints;
    unsigned _numPhases;
};

#endif