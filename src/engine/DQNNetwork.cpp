#include "DQNNetwork.h"

QNetwork::QNetwork( const unsigned numPlConstraints,
                    unsigned numFeatures,
                    const unsigned numActions )
    :
     _outputDim( numActions ), _numFeatures( numFeatures ) ,_numConstraints( numPlConstraints )
{
    _inputDim = numPlConstraints * _numFeatures;
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
    const auto features = stateWithBatch.narrow( 2, 0, _numFeatures ).to( torch::kFloat32 );
    auto featuresFlattened = features.view( { features.size( 0 ), -1 } );
    if ( featuresFlattened.sizes().size() != 2 || featuresFlattened.size( 1 ) != fc1->options.in_features() )
    {
        std::cerr << "Invalid input tensor size: Expected [batch_size, "
                  << fc1->options.in_features() << "], got " << featuresFlattened.sizes() << std::endl;
        throw std::runtime_error( "Invalid input tensor size." );
    }

    if ( torch::isnan( featuresFlattened ).any().item<bool>() ||
         torch::isinf( featuresFlattened ).any().item<bool>() )
    {
        std::cerr << "Error: fullInput contains NaN or Inf values!" << std::endl;
        throw std::runtime_error( "NaN/Inf detected in fullInput." );
    }
    auto x = torch::relu( fc1( featuresFlattened ) );
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
    printf( "%s - Bias norm: %f\n ", name.c_str(), bias.norm().item<float>() );
    fflush( stdout );
}

std::pair<int, int> QNetwork::getDims() const
{
    return { _inputDim, _outputDim };
}

void QNetwork::save( torch::serialize::OutputArchive &archive ) const
{
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