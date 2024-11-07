#ifndef DQNSTATE_H
#define DQNSTATE_H
#include <torch/torch.h>

class State {
public:
    State(int numConstraints, int defaultPhase);
    torch::Tensor toTensor() const;
    int encodeStateIndex(const std::pair<int, int>& element) const;
    void updateState(int constraintIndex, int newPhase);
    const std::vector<std::pair<int, int>>& getData() const;
private:
    std::vector<std::pair<int, int>> _stateData;
    unsigned _numPhases;
};
#endif //DQNSTATE_H
