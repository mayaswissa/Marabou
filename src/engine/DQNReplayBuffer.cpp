#include "DQNReplayBuffer.h"
#include <random>

ReplayBuffer::ReplayBuffer(const unsigned action_size, const unsigned buffer_size, const unsigned batch_size )
    : action_size_(action_size), buffer_size_(buffer_size), batch_size_(batch_size) {
}

void ReplayBuffer::add(const torch::Tensor& state, const torch::Tensor& action, unsigned reward, const torch::Tensor& next_state, bool done) {
    if (memory_.size() >= buffer_size_) {
        memory_.pop_front();
    }
    memory_.emplace_back(state, action, reward, next_state, done);
}

std::vector<Experience> ReplayBuffer::sample() const
{
    std::vector<Experience> experiences;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size() - 1);

    // Sample `batch_size_` experiences randomly
    for (size_t i = 0; i < batch_size_; ++i) {
        experiences.push_back(memory_[dis(gen)]);
    }

    return experiences;
}

size_t ReplayBuffer::size() const {
    return memory_.size();
}