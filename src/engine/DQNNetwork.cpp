#include "DQNNetwork.h"

QNetwork::QNetwork(unsigned state_size, unsigned action_size, unsigned fc1_size, unsigned fc2_size) {
    fc1 = register_module("fc1", torch::nn::Linear(state_size, fc1_size));
    fc2 = register_module("fc2", torch::nn::Linear(fc1_size, fc2_size));
    fc3 = register_module("fc3", torch::nn::Linear(fc2_size, action_size));
}

torch::Tensor QNetwork::forward(torch::Tensor state) {
    if (state.sizes().size() != 2 || state.size(1) != fc1->options.in_features()) {
        std::cerr << "Invalid state size: Expected [batch_size, " << fc1->options.in_features() << "], got " << state.sizes() << std::endl;
        throw std::runtime_error("Invalid state size received by QNetwork");
    }
    state = torch::relu(fc1(state));
    state = torch::relu(fc2(state));
    return fc3(state);
}

std::vector<torch::Tensor> QNetwork::get_parameters() const
{
    return this->parameters();
}
