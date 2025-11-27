#ifndef QNETWORK_H
#define QNETWORK_H
#undef Warning
#include <torch/torch.h>

class QNetwork final : public torch::nn::Module
{
public:
    QNetwork(unsigned numConstraints,
             unsigned numLocalFeatures,
             unsigned numActions,
             unsigned numGlobalFeatures);
    torch::Tensor forward( const torch::Tensor &state );
    std::vector<torch::Tensor> getParameters() const;
    std::pair<int, int> getDims() const;
    void save( torch::serialize::OutputArchive &archive ) const;
    void load( torch::serialize::InputArchive &archive );

private:
    void initWeights();
    unsigned _numConstraints, _numLocalFeatures, _numGlobalFeatures;
    unsigned _inputDim, _outputDim;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Linear fcAdv1{nullptr}, fcAdv2{nullptr};
    torch::nn::Linear fcVal1{nullptr}, fcVal2{nullptr};
};
#endif
