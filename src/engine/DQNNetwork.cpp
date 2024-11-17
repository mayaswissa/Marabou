#include "DQNNetwork.h"

QNetwork::QNetwork( unsigned numPlConstraints,
                    unsigned numPhaseStatuses,
                    unsigned embeddingDim,
                    unsigned numActions )
    : _statusEmbedding( register_module( "statusEmbedding",
                                         torch::nn::Embedding( numPhaseStatuses, embeddingDim ) ) )
{
    unsigned inputDim = numPlConstraints * numPhaseStatuses * embeddingDim;
    std::cout << "Input Dimension: " << inputDim << std::endl;
    std::cout << "Number of Constraints: " << numPlConstraints << std::endl;
    std::cout << " numPhaseStatuses: " << numPhaseStatuses << std::endl;
    std::cout << "Embedding Dimension: " << embeddingDim << std::endl;


    fc1 = register_module( "fc1", torch::nn::Linear( inputDim, 64 ) );
    fc2 = register_module( "fc2", torch::nn::Linear( 64, 128 ) );
    fc3 = register_module( "fc3", torch::nn::Linear( 128, numActions ) );
}

torch::Tensor QNetwork::forward( const torch::Tensor &state )
{
    auto indexTensor = state.to(torch::kInt64);
    std::cout << "State tensor type: " << indexTensor.dtype() << std::endl;
    std::cout << "State tensor sizes: " << indexTensor.sizes() << std::endl;
    // Applying the embedding layer.
    auto embedded = _statusEmbedding->forward( indexTensor );
    auto flattened = embedded.view({embedded.size(0), -1});
    std::cout << "After flattening: size = " << flattened.sizes() << std::endl;

    // Check if the embedded tensor has the expected dimensions before applying the first linear
    // layer.
    if ( flattened.sizes().size() != 2 || flattened.size( 1 ) != fc1->options.in_features() )
    {
        std::cerr << "Invalid embedded state size: Expected [batch_size, "
                  << fc1->options.in_features() << "], got " << flattened.sizes() << std::endl;
        throw std::runtime_error( "Invalid embedded state size received by QNetwork" );
    }

    // Processing through the first fully connected layer.
    auto x = torch::relu( fc1( flattened ) );
    std::cout << "After first FC layer: size = " << x.sizes() << std::endl;

    // Processing through the second fully connected layer.
    x = torch::relu( fc2( x ) );
    std::cout << "After second FC layer: size = " << x.sizes() << std::endl;

    // Final output from the network - Q vales of each action in action space.
    auto output = fc3( x );
    std::cout << "Output size: " << output.sizes() << std::endl;

    return output;
}


std::vector<torch::Tensor> QNetwork::getParameters() const
{
    return this->parameters();
}
