#include "DQNState.h"

State::State(int numConstraints, int numPhases)
    : _numPhases(numPhases), _stateData(numConstraints, std::vector<int>(numPhases, 0)) {
    for (int i = 0; i < numConstraints; ++i) {
        _stateData[i][0] = 1;
    }
}

torch::Tensor State::toTensor() const {
    std::vector<float> flatData;
    for (const auto& constraint : _stateData) {
        for (int phase : constraint) {
            flatData.push_back(static_cast<float>(phase));
        }
    }
    return torch::tensor(flatData).view({ static_cast<long>( _stateData.size() ), _numPhases});
}

void State::updateState(int constraintIndex, int newPhase) {
    if (constraintIndex < _stateData.size() && newPhase < _numPhases) {
        std::fill(_stateData[constraintIndex].begin(), _stateData[constraintIndex].end(), 0);
        _stateData[constraintIndex][newPhase] = 1;
    }
}

const std::vector<std::vector<int>>& State::getData() const {
    return _stateData;
}

int State::encodeStateIndex(const std::pair<int, int>& element) const {
    return element.first * _numPhases + element.second;
}