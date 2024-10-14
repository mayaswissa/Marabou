#ifndef QNETWORK_H
#define QNETWORK_H
#undef Warning
#include <torch/torch.h>

class QNetwork : public torch::nn::Module {
public:
    QNetwork(int64_t state_size, int64_t action_size, int64_t fc1_size = 64, int64_t fc2_size = 64);
    torch::Tensor forward( torch::Tensor state );
    std::vector<torch::Tensor> get_parameters() const;

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif