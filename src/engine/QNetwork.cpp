#include "QNetwork.h"

QNetwork::QNetwork(int64_t state_size, int64_t action_size, int64_t fc1_size, int64_t fc2_size) {
    fc1 = register_module("fc1", torch::nn::Linear(state_size, fc1_size));
    fc2 = register_module("fc2", torch::nn::Linear(fc1_size, fc2_size));
    fc3 = register_module("fc3", torch::nn::Linear(fc2_size, action_size));
}

torch::Tensor QNetwork::forward(torch::Tensor state) {
    state = torch::relu(fc1(state));
    state = torch::relu(fc2(state));
    return fc3(state);
}

std::vector<torch::Tensor> QNetwork::get_parameters() const
{
    return this->parameters();
}
