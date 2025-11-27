#include "DQNState.h"

#include <cmath>
#include <limits>

namespace {
inline float squashFeature( const double x, double s = 10.0 )
{
    if ( std::isnan( x ) )
        return 0.0f;
    if ( std::isinf( x ) )
        return std::signbit( x ) ? -1.0f : 1.0f;
    return static_cast<float>( std::tanh( x / s ) );
}
}

State::State( const unsigned numConstraints )
    : _stateData()
    , _numConstraints( numConstraints )
    , _numPhases( DQN_NUM_PHASES )
{
    _stateData.assign( numConstraints * TOTAL_FEATURES, 0.0f );
    for ( unsigned i = 0; i < numConstraints; ++i )
        _stateData[i * TOTAL_FEATURES + DQN_RELU_NOT_FIXED_VALUE] = 1.0f;
}

State::State( const State &other )
    : _stateData( other._stateData )
    , _numConstraints( other._numConstraints )
    , _numPhases( other._numPhases )
{
}

State &State::operator=( const State &other )
{
    if ( this == &other )
        return *this;

    _stateData = other._stateData;
    _numConstraints = other._numConstraints;
    _numPhases = other._numPhases;
    return *this;
}

torch::Tensor State::toTensor() const
{
    auto stateTensor = torch::from_blob( const_cast<float *>( _stateData.data() ),
                                         { static_cast<long>( _numConstraints ),
                                           static_cast<long>( TOTAL_FEATURES ) },
                                         torch::TensorOptions().dtype( torch::kFloat32 ) )
                           .clone();
    return stateTensor;
}

void State::updateConstraintPhase( const unsigned constraintIndex, const unsigned newPhase )
{
    if ( constraintIndex >= _numConstraints || newPhase >= DQN_NUM_PHASES || _stateData.empty() )
        return;

    const size_t rowStart = static_cast<size_t>( constraintIndex ) * TOTAL_FEATURES;
    for ( unsigned k = 0; k < DQN_NUM_PHASES; ++k )
        _stateData[rowStart + DQN_RELU_NOT_FIXED_VALUE + k] = 0.0f;
    _stateData[rowStart + DQN_RELU_NOT_FIXED_VALUE + newPhase] = 1.0f;
}

void State::updateBounds( const unsigned constraintIndex,
                          const double upperBound,
                          const double lowerBound )
{
    if ( constraintIndex >= _numConstraints )
        return;
    const size_t base = static_cast<size_t>( constraintIndex ) * TOTAL_FEATURES;
    _stateData[base + DQN_RELU_LOWER_BOUND] = squashFeature( lowerBound, 30 );
    _stateData[base + DQN_RELU_UPPER_BOUND] = squashFeature( upperBound, 30 );
}

void State::updateSoIScoreForAgent( const unsigned constraintIndex,
                                    const double SoiActiveScore,
                                    const double SoiInactiveScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    const size_t base = static_cast<size_t>( constraintIndex ) * TOTAL_FEATURES;
    _stateData[base + SOI_ACTIVE_SCORE] = squashFeature( SoiActiveScore );
    _stateData[base + SOI_INACTIVE_SCORE] = squashFeature( SoiInactiveScore );
}

void State::updatePolarity( const unsigned constraintIndex, const double polarityScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[static_cast<size_t>( constraintIndex ) * TOTAL_FEATURES + POLARITY_SCORE] =
        squashFeature( polarityScore );
}

void State::updateBaBsrScore( const unsigned constraintIndex, const double BaBsrScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[static_cast<size_t>( constraintIndex ) * TOTAL_FEATURES + BaBsr_SCORE] =
        squashFeature( BaBsrScore );
}

void State::updateGlobalFeatures( unsigned unstableCount, unsigned treeDepth, unsigned splitsSoFar )
{
    for ( unsigned i = 0; i < _numConstraints; ++i )
    {
        const size_t base = static_cast<size_t>( i ) * TOTAL_FEATURES;
        _stateData[base + NUM_LOCAL_FEATURES + GF_UNSTABLE_COUNT] =
            static_cast<float>( unstableCount );
        _stateData[base + NUM_LOCAL_FEATURES + GF_TREE_DEPTH] = static_cast<float>( treeDepth );
        _stateData[base + NUM_LOCAL_FEATURES + GF_SPLITS_SO_FAR] =
            static_cast<float>( splitsSoFar );
    }
}
