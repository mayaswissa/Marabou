#include "DQNNetwork.h"

QNetwork::QNetwork( unsigned numPlConstraints,
                    unsigned numPhaseStatuses,
                    unsigned embeddingDim,
                    unsigned numActions )
    : _statusEmbedding( register_module( "statusEmbedding",
                                         torch::nn::Embedding( numPhaseStatuses, embeddingDim ) ) )
    , dropout( register_module( "dropout", torch::nn::Dropout( 0.5 ) ) )
{
    unsigned inputDim = numPlConstraints * numPhaseStatuses * embeddingDim;
    fc1 = register_module( "fc1", torch::nn::Linear( inputDim, 64 ) );
    fc2 = register_module( "fc2", torch::nn::Linear( 64, 128 ) );
    fc3 = register_module( "fc3", torch::nn::Linear( 128, numActions ) );
    initWeights();
}

void QNetwork::initWeights()
{
    // Initialize weights
    torch::nn::init::kaiming_normal_( fc1->weight, 0.0, torch::kFanOut, torch::kReLU );
    torch::nn::init::kaiming_normal_( fc2->weight, 0.0, torch::kFanOut, torch::kReLU );
    torch::nn::init::kaiming_normal_( fc3->weight, 0.0, torch::kFanOut, torch::kReLU );

    // Initialize biases to zero if biases are used
    if ( fc1->bias.defined() )
        torch::nn::init::constant_( fc1->bias, 0.0 );
    if ( fc2->bias.defined() )
        torch::nn::init::constant_( fc2->bias, 0.0 );
    if ( fc3->bias.defined() )
        torch::nn::init::constant_( fc3->bias, 0.0 );
}

torch::Tensor QNetwork::forward( const torch::Tensor &state )
{
    auto indexTensor = state.to( torch::kInt64 );

    // Validating state indices for embedding
    if ( indexTensor.any().item<bool>() &&
         ( indexTensor.min().item<int>() < 0 ||
           indexTensor.max().item<int>() >= _statusEmbedding->options.num_embeddings() ) )
    {
        std::cerr << "IndexTensor out of bounds: Min " << indexTensor.min().item<int>() << ", Max "
                  << indexTensor.max().item<int>() << std::endl;
        throw std::runtime_error( "State index out of embedding bounds." );
    }

    // Applying the embedding layer
    auto embedded = _statusEmbedding->forward( indexTensor );
    auto flattened = embedded.view( { embedded.size( 0 ), -1 } );

    // Dimension check for neural network input
    if ( flattened.sizes().size() != 2 || flattened.size( 1 ) != fc1->options.in_features() )
    {
        std::cerr << "Invalid embedded state size: Expected [batch_size, "
                  << fc1->options.in_features() << "], got " << flattened.sizes() << std::endl;
        throw std::runtime_error( "Invalid embedded state size received by QNetwork" );
    }

    auto x = torch::relu( fc1( flattened ) );
    // x = dropout(x);
    x = torch::relu( fc2( x ) );
    // x = dropout(x);
    // auto output = torch::tanh(fc3(x)) * 10;
    auto output = fc3( x );
    return output;
}

std::vector<torch::Tensor> QNetwork::getParameters() const
{
    return this->parameters();
}
