#include "DQNState.h"

State::State(int numConstraints, int defaultPhase)
        : _stateData(numConstraints, std::make_pair(0, defaultPhase)) {
    // Initialize each constraint with a default phase
    for (int i = 0; i < numConstraints; ++i) {
        _stateData[i].first = i; // Set constraint index
    }
}
torch::Tensor State::toTensor() const {
    std::vector<int> indices;
    for (const auto& element : _stateData) {
        int index = encodeStateIndex(element);
        indices.push_back(index);
    }
    return torch::tensor(indices, torch::dtype(torch::kInt64));
}
const std::vector<std::pair<int, int>>& State::getData() const {
    return _stateData;
}

int State::encodeStateIndex(const std::pair<int, int>& element) const {
    return element.first * _numPhases + element.second;
}
void State::updateState(int constraintIndex, int newPhase) {
    for (auto& element : _stateData) {
        if (element.first == constraintIndex) {
            element.second = newPhase;
            break;
        }
    }
}
