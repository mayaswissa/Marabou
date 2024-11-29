#include "DQNReplayBuffer.h"
#include <random>

void Experience::updateReward(double newReward)
{
    this->reward = newReward;
}

ReplayBuffer::ReplayBuffer(const unsigned actionSize, const unsigned bufferSize, const unsigned batchSize )
    : _actionSize(actionSize), _bufferSize(bufferSize), _batchSize(batchSize) {
}

void ReplayBuffer::add(const torch::Tensor& state, const torch::Tensor& action, float reward, const torch::Tensor& nextState, bool done) {
    if (_memory.size() >= _bufferSize) {
        _memory.popFirst();
    }
    _memory.append(std::make_unique<Experience>(state, action, reward, nextState, done));
}
void ReplayBuffer::add( std::unique_ptr<Experience> experience ) {
    if (_memory.size() >= _bufferSize) {
        _memory.popFirst();
    }
    _memory.append(std::move(experience));
}


Vector<std::unique_ptr<Experience>> ReplayBuffer::sample() const{
    Vector<std::unique_ptr<Experience>> sampledExperiences;

    if (_memory.empty() || _batchSize == 0) {
        return sampledExperiences;
    }

    // Determine the number of experiences to sample
    size_t sampleSize = std::min(_batchSize, _memory.size());

    // Shuffle the deque to randomize selection
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(_memory.begin(), _memory.end(), g);

    // Move selected experiences to the output vector and erase them from the deque
    for (size_t i = 0; i < sampleSize; ++i) {
        sampledExperiences.append(std::move(_memory.last()));
        _memory.pop();
    }

    return sampledExperiences;
}

size_t ReplayBuffer::size() const {
    return _memory.size();
}