#ifndef QNETWORK_H
#define QNETWORK_H
#undef Warning
#include <torch/torch.h>

class QNetwork final : public torch::nn::Module {
public:
    QNetwork(unsigned numVariables, unsigned numPhaseStatuses, unsigned embeddingDim, unsigned numActions);
    torch::Tensor forward( torch::Tensor state );
    std::vector<torch::Tensor> getParameters() const;
private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif