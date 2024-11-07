#include "DQNNetwork.h"

QNetwork::QNetwork( unsigned numVariables,
                    unsigned numPhaseStatuses,
                    unsigned embeddingDim,
                    unsigned numActions )
{
    auto statusEmbedding = register_module(
        "statusEmbedding", torch::nn::Embedding( numPhaseStatuses, embeddingDim ) );
    unsigned inputDim = numVariables * embeddingDim;
    fc1 = register_module( "fc1", torch::nn::Linear( inputDim, 64 ) );
    fc2 = register_module( "fc2", torch::nn::Linear( 64, 128 ) );
    fc3 = register_module( "fc3", torch::nn::Linear( 128, numActions ) );
}

torch::Tensor QNetwork::forward( torch::Tensor state )
{
    if ( state.sizes().size() != 2 || state.size( 1 ) != fc1->options.in_features() )
    {
        std::cerr << "Invalid state size: Expected [batch_size, " << fc1->options.in_features()
                  << "], got " << state.sizes() << std::endl;
        throw std::runtime_error( "Invalid state size received by QNetwork" );
    }
    state = torch::relu( fc1( state ) );
    state = torch::relu( fc2( state ) );
    return fc3( state );
}

std::vector<torch::Tensor> QNetwork::getParameters() const
{
    return this->parameters();
}
