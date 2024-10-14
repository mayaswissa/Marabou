#ifndef REPLAYBUFFER_H
#define REPLAYBUFFER_H

#include <torch/torch.h>
#include <deque>
#include <vector>

class ReplayBuffer {
public:
    ReplayBuffer(int64_t action_size, size_t buffer_size, size_t batch_size);
    void add(const torch::Tensor& state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, torch::Tensor done);
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> sample() const;
    size_t size() const;

private:
    int64_t action_size_;
    size_t buffer_size_;
    size_t batch_size_;
    std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory_;
};

#endif