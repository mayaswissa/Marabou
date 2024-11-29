#include "DQNReplayBuffer.h"
#include <random>

ReplayBuffer::ReplayBuffer(const unsigned actionSize, const unsigned bufferSize, const unsigned batchSize )
    : _actionSize(actionSize), _bufferSize(bufferSize), _batchSize(batchSize) {
}

void ReplayBuffer::add(const torch::Tensor& state, const torch::Tensor& action, float reward, const torch::Tensor& nextState, bool done) {
    if (_memory.size() >= _bufferSize) {
        _memory.pop_front();
    }
    _memory.emplace_back(state, action, reward, nextState, done);
}
void ReplayBuffer::add(Experience experience) {
    if (_memory.size() >= _bufferSize) {
        _memory.pop_front();
    }
    _memory.emplace_back(experience);
}


Vector<Experience> ReplayBuffer::sample() const
{
    Vector<Experience> experiences;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size() - 1);
    // Sample `batch_size_` experiences randomly
    for (size_t i = 0; i < _batchSize; ++i) {
        experiences.append(_memory[dis(gen)]);
    }

    return experiences;
}

size_t ReplayBuffer::size() const {
    return _memory.size();
}