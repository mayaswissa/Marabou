#ifndef QNETWORK_H
#define QNETWORK_H
#undef Warning
#include <torch/torch.h>

class QNetwork final : public torch::nn::Module {
public:
    QNetwork(unsigned state_size, unsigned action_size, unsigned fc1_size = 64, unsigned fc2_size = 64);
    torch::Tensor forward( torch::Tensor state );
    std::vector<torch::Tensor> get_parameters() const;
private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif