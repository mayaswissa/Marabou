#include "DQNNetwork.h"

QNetwork::QNetwork( const unsigned numPlConstraints,
                    unsigned numPhases,
                    unsigned embeddingDim,
                    const unsigned numActions )
    : _statusEmbedding(
          register_module( "statusEmbedding", torch::nn::Embedding( numPhases, embeddingDim ) ) )
    // , dropout( register_module( "dropout", torch::nn::Dropout( 0.3 ) ) )
    , _numPhases( numPhases )
    , _embeddingDim( embeddingDim )
    , _numBounds( 2 )
{
    _inputDim = numPlConstraints * ( _numPhases * embeddingDim + _numBounds );
    _outputDim = numActions;
    _numConstraints = numPlConstraints;
    fc1 = register_module( "fc1", torch::nn::Linear( _inputDim, 64 ) );
    fc2 = register_module( "fc2", torch::nn::Linear( 64, 128 ) );
    fc3 = register_module( "fc3", torch::nn::Linear( 128, 256 ) );
    fc4 = register_module( "fc4", torch::nn::Linear( 256, _outputDim ) );
    initWeights();
}

void QNetwork::initWeights()
{
    // Initialize weights
    torch::nn::init::kaiming_normal_( fc1->weight, 0.0, torch::kFanOut, torch::kReLU );
    torch::nn::init::kaiming_normal_( fc2->weight, 0.0, torch::kFanOut, torch::kReLU );
    torch::nn::init::kaiming_normal_( fc3->weight, 0.0, torch::kFanOut, torch::kReLU );
    torch::nn::init::kaiming_normal_( fc4->weight, 0.0, torch::kFanOut, torch::kReLU );

    // Initialize biases to zero if biases are used
    if ( fc1->bias.defined() )
        torch::nn::init::constant_( fc1->bias, 0.0 );
    if ( fc2->bias.defined() )
        torch::nn::init::constant_( fc2->bias, 0.0 );
    if ( fc3->bias.defined() )
        torch::nn::init::constant_( fc3->bias, 0.0 );
    if ( fc4->bias.defined() )
        torch::nn::init::constant_( fc4->bias, 0.0 );
}

torch::Tensor QNetwork::forward( const torch::Tensor &state )
{
    auto stateWithBatch = state.to( torch::kFloat32 );
    if ( state.sizes().size() == 2 )
        stateWithBatch = state.unsqueeze( 0 ); // Add batch dimension

    const auto phases = stateWithBatch.narrow( 2, 0, _numPhases ).to( torch::kInt64 );
    const auto bounds = stateWithBatch.narrow( 2, _numPhases, _numBounds ).to( torch::kFloat64 );

    if ( phases.any().item<bool>() &&
         ( phases.min().item<int>() < 0 ||
           phases.max().item<int>() >=
               static_cast<int>( _statusEmbedding->options.num_embeddings() ) ) )
    {
        std::cerr << "Phase index out of bounds: Min " << phases.min().item<int>() << ", Max "
                  << phases.max().item<int>() << " vs num_embeddings "
                  << _statusEmbedding->options.num_embeddings() << std::endl;
        throw std::runtime_error( "Phase index out of embedding bounds." );
    }

    // Applying the embedding layer
    auto embedded = _statusEmbedding->forward( phases );
    auto embeddedFlattened = embedded.view( { embedded.size( 0 ), -1 } );

    auto boundsFlattened = bounds.view( { bounds.size( 0 ), -1 } );
    auto fullInput = torch::cat( { embeddedFlattened, boundsFlattened }, 1 );
    fullInput = fullInput.to( torch::kFloat32 );

    if ( fullInput.sizes().size() != 2 || fullInput.size( 1 ) != fc1->options.in_features() )
    {
        std::cerr << "Invalid input tensor size: Expected [batch_size, "
                  << fc1->options.in_features() << "], got " << fullInput.sizes() << std::endl;
        throw std::runtime_error( "Invalid input tensor size." );
    }

    if ( torch::isnan( fullInput ).any().item<bool>() ||
         torch::isinf( fullInput ).any().item<bool>() )
    {
        std::cerr << "Error: fullInput contains NaN or Inf values!" << std::endl;
        throw std::runtime_error( "NaN/Inf detected in fullInput." );
    }

    auto x = torch::relu( fc1( fullInput ) );
    x = torch::relu( fc2( x ) );
    x = torch::relu( fc3( x ) );
    auto output = fc4( x );
    // If the input was a single state, remove batch dimension from output
    if ( state.sizes().size() == 2 )
    {
        output = output.squeeze( 0 );
    }
    return output;
}

std::vector<torch::Tensor> QNetwork::getParameters() const
{
    return this->parameters();
}
void check_weights( const torch::nn::Linear &layer, const std::string &name )
{
    auto weights = layer->weight;
    auto bias = layer->bias;

    printf( "new weights: \n" );
    printf( "%s - Weight norm: %f\n ", name.c_str(), weights.norm().item<float>() );
    printf( "%s - Bias norm: %f\n ", name.c_str(), weights.norm().item<float>() );
    fflush( stdout );
}

std::pair<int, int> QNetwork::getDims() const
{
    return { _inputDim, _outputDim };
}

void QNetwork::save( torch::serialize::OutputArchive &archive ) const
{
    // Save weights and biases of the embedding and linear layers
    archive.write( "statusEmbedding_weight", _statusEmbedding->weight );
    // Save weights and biases for each Linear layer
    archive.write( "fc1_weight", fc1->weight );
    archive.write( "fc1_bias", fc1->bias );
    archive.write( "fc2_weight", fc2->weight );
    archive.write( "fc2_bias", fc2->bias );
    archive.write( "fc3_weight", fc3->weight );
    archive.write( "fc3_bias", fc3->bias );
    archive.write( "fc4_weight", fc4->weight );
    archive.write( "fc4_bias", fc4->bias );
    // check_weights(fc1, "FC1");
    // check_weights(fc2, "FC2");
    // check_weights(fc3, "FC3");
}


void QNetwork::load( torch::serialize::InputArchive &archive )
{
    // Load weights and biases of the embedding and linear layers
    archive.read( "statusEmbedding_weight", _statusEmbedding->weight );
    // Load weights and biases for each Linear layer
    archive.read( "fc1_weight", fc1->weight );
    archive.read( "fc1_bias", fc1->bias );
    archive.read( "fc2_weight", fc2->weight );
    archive.read( "fc2_bias", fc2->bias );
    archive.read( "fc3_weight", fc3->weight );
    archive.read( "fc3_bias", fc3->bias );
    archive.read( "fc4_weight", fc4->weight );
    archive.read( "fc4_bias", fc4->bias );
    // check_weights(fc1, "FC1");
    // check_weights(fc2, "FC2");
    // check_weights(fc3, "FC3");
}