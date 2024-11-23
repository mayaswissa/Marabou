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
    float reward;
    torch::Tensor nextState;
    bool done;

    Experience(torch::Tensor  state, const torch::Tensor& action,
                const float reward, const torch::Tensor& nextState,
                const bool done)
        : state(std::move(state)), action(action), reward(reward), nextState(nextState), done(done) {}
};


class ReplayBuffer {
public:
    ReplayBuffer(unsigned actionSize, unsigned bufferSize, unsigned batchSize);
    void add(const torch::Tensor& state, const torch::Tensor& action, float reward, const torch::Tensor& nextState, bool done);

    std::vector<Experience> sample() const;
    size_t size() const;

private:
    unsigned _actionSize;
    unsigned _bufferSize;
    unsigned _batchSize;
    std::deque<Experience> _memory;
};

#endif