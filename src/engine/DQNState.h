#ifndef DQNSTATE_H
#define DQNSTATE_H

#include <vector>
#include <torch/torch.h>

class State {
public:
    State(int numConstraints, int numPhases);

    torch::Tensor toTensor() const;
    void updateState(int constraintIndex, int newPhase);
    int encodeStateIndex(const std::pair<int, int>& element) const;
    const std::vector<std::vector<int>>& getData() const;

private:
    std::vector<std::vector<int>> _stateData; // One-hot encoded state data
    int _numPhases;
};

#endif // DQNSTATE_H