#ifndef DQNREPLAYBUFFER_H
#define DQNREPLAYBUFFER_H
#include <deque>
#include <utility>
#include <vector>
#undef Warning
#include <torch/torch.h>

struct Experience {
    torch::Tensor state;
    torch::Tensor action;
    unsigned reward;
    torch::Tensor nextState;
    bool done;

    Experience(torch::Tensor  state, const torch::Tensor& action,
                const unsigned reward, const torch::Tensor& nextState,
                const bool done)
        : state(std::move(state)), action(action), reward(reward), nextState(nextState), done(done) {}
};


class ReplayBuffer {
public:
    ReplayBuffer(unsigned action_size, unsigned buffer_size, unsigned batch_size);
    void add(const torch::Tensor& state, const torch::Tensor& action, unsigned reward, const torch::Tensor& next_state, bool done);

    std::vector<Experience> sample() const;
    size_t size() const;

private:
    unsigned action_size_;
    unsigned buffer_size_;
    unsigned batch_size_;
    std::deque<Experience> memory_;
};

#endif