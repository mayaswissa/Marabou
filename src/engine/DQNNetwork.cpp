#include "DQNNetwork.h"

QNetwork::QNetwork( const unsigned numConstraints,
                    unsigned numLocalFeatures,
                    unsigned numActions,
                    unsigned numGlobalFeatures )
    : _numConstraints( numConstraints )
    , _numLocalFeatures( numLocalFeatures )
    , _numGlobalFeatures( numGlobalFeatures )
    , _outputDim( numActions )
{
    _inputDim = numConstraints * numLocalFeatures + numGlobalFeatures;
    fc1 = register_module( "fc1", torch::nn::Linear( _inputDim, 128 ) );
    fc2 = register_module( "fc2", torch::nn::Linear( 128, 128 ) );
    fcAdv1 = register_module( "adv1", torch::nn::Linear( 128, 64 ) );
    fcAdv2 = register_module( "adv2", torch::nn::Linear( 64, _outputDim ) );
    fcVal1 = register_module( "val1", torch::nn::Linear( 128, 64 ) );
    fcVal2 = register_module( "val2", torch::nn::Linear( 64, 1 ) );

    initWeights();
}


void QNetwork::initWeights()
{
    for ( auto *l : { fc1.ptr().get(),
                      fc2.ptr().get(),
                      fcAdv1.ptr().get(),
                      fcAdv2.ptr().get(),
                      fcVal1.ptr().get(),
                      fcVal2.ptr().get() } )
    {
        torch::nn::init::kaiming_normal_( l->weight, 0.0, torch::kFanOut, torch::kReLU );
        if ( l->bias.defined() )
            torch::nn::init::constant_( l->bias, 0.0f );
    }
}

torch::Tensor QNetwork::forward( const torch::Tensor &state )
{
    auto x = state.to( torch::kFloat32 );
    x = torch::nan_to_num( x, 0.0, 0.0, 0.0 );
    if ( x.dim() == 2 )
        x = x.unsqueeze( 0 );
    auto B = x.size( 0 );

    auto global = x.index( { torch::indexing::Slice(),
                             0,
                             torch::indexing::Slice( _numLocalFeatures,
                                                     _numLocalFeatures + _numGlobalFeatures ) } )
                      .reshape( { B, (long)_numGlobalFeatures } );

    auto local = x.index( { torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice( 0, _numLocalFeatures ) } )
                     .contiguous()
                     .view( { B, (long)( _numConstraints * _numLocalFeatures ) } );

    auto input = torch::cat( { local, global }, 1 );

    auto h = torch::relu( fc1->forward( input ) );
    h = torch::relu( fc2->forward( h ) );

    auto a = torch::relu( fcAdv1->forward( h ) );
    a = fcAdv2->forward( a );

    auto v = torch::relu( fcVal1->forward( h ) );
    v = fcVal2->forward( v );

    auto a_mean = a.mean( 1, true );
    return v + ( a - a_mean );
}

std::vector<torch::Tensor> QNetwork::getParameters() const
{
    return this->parameters();
}

std::pair<int, int> QNetwork::getDims() const
{
    return { static_cast<int>( _inputDim ), static_cast<int>( _outputDim ) };
}

void QNetwork::save( torch::serialize::OutputArchive &archive ) const
{
    archive.write( "fc1_weight", fc1->weight );
    archive.write( "fc1_bias", fc1->bias );
    archive.write( "fc2_weight", fc2->weight );
    archive.write( "fc2_bias", fc2->bias );
    archive.write( "fcAdv1_weight", fcAdv1->weight );
    archive.write( "fcAdv1_bias", fcAdv1->bias );
    archive.write( "fcAdv2_weight", fcAdv2->weight );
    archive.write( "fcAdv2_bias", fcAdv2->bias );
    archive.write( "fcVal1_weight", fcVal1->weight );
    archive.write( "fcVal1_bias", fcVal1->bias );
    archive.write( "fcVal2_weight", fcVal2->weight );
    archive.write( "fcVal2_bias", fcVal2->bias );
}

void QNetwork::load( torch::serialize::InputArchive &archive )
{
    archive.read( "fc1_weight", fc1->weight );
    archive.read( "fc1_bias", fc1->bias );
    archive.read( "fc2_weight", fc2->weight );
    archive.read( "fc2_bias", fc2->bias );
    archive.read( "fcAdv1_weight", fcAdv1->weight );
    archive.read( "fcAdv1_bias", fcAdv1->bias );
    archive.read( "fcAdv2_weight", fcAdv2->weight );
    archive.read( "fcAdv2_bias", fcAdv2->bias );
    archive.read( "fcVal1_weight", fcVal1->weight );
    archive.read( "fcVal1_bias", fcVal1->bias );
    archive.read( "fcVal2_weight", fcVal2->weight );
    archive.read( "fcVal2_bias", fcVal2->bias );
}