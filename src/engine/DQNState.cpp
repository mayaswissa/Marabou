#include "DQNState.h"

State::State( const unsigned numConstraints )
    : _stateData( numConstraints )
    , _numConstraints( numConstraints )
    , _numPhases( DQN_NUM_PHASES )
{
    // Allocate a contiguous vector with (numConstraints * NUM_FEATURES) elements.
    _stateData.resize( numConstraints * TOTAL_FEATURES, 0.0 );

    // For each constraint, set the feature at index DQN_RELU_NOT_FIXED to 1.0,
    for ( unsigned i = 0; i < numConstraints; ++i )
        _stateData[i * TOTAL_FEATURES + DQN_RELU_NOT_FIXED_VALUE] = 1.0;
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
    auto stateTensor =
        torch::tensor( _stateData, torch::dtype( torch::kFloat32 ) )
            .view( { static_cast<long>( _numConstraints ), static_cast<long>( TOTAL_FEATURES ) } );

    constexpr float INF_CAP = 1e9f;
    stateTensor.masked_fill_( stateTensor == std::numeric_limits<float>::infinity(), INF_CAP );
    stateTensor.masked_fill_( stateTensor == -std::numeric_limits<float>::infinity(), -INF_CAP );

    if ( !torch::isfinite( stateTensor ).all().to( torch::kCPU ).item<bool>() )
        throw std::runtime_error( "Non-finite features after State::toTensor()" );

    return stateTensor;
}

void State::updateConstraintPhase( const unsigned constraintIndex, const unsigned newPhase )
{
    if ( constraintIndex >= _numConstraints || newPhase >= DQN_NUM_PHASES || _stateData.empty() )
        return;

    size_t rowStart = constraintIndex * TOTAL_FEATURES;
    for ( unsigned k = 0; k < DQN_NUM_PHASES; ++k )
        _stateData[rowStart + DQN_RELU_NOT_FIXED_VALUE + k] = 0.0;
    _stateData[rowStart + DQN_RELU_NOT_FIXED_VALUE + newPhase] = 1.0;
}
void State::updateBounds( const unsigned constraintIndex,
                          const double upperBound,
                          const double lowerBound )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * TOTAL_FEATURES + DQN_RELU_LOWER_BOUND] = lowerBound;
    _stateData[constraintIndex * TOTAL_FEATURES + DQN_RELU_UPPER_BOUND] = upperBound;
}

void State::updateSoIScoreForAgent( const unsigned constraintIndex,
                                    const double SoiActiveScore,
                                    const double SoiInactiveScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * TOTAL_FEATURES + SOI_ACTIVE_SCORE] = SoiActiveScore;
    _stateData[constraintIndex * TOTAL_FEATURES + SOI_INACTIVE_SCORE] = SoiInactiveScore;
}

void State::updatePolarity( const unsigned constraintIndex, const double polarityScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * TOTAL_FEATURES + POLARITY_SCORE] = polarityScore;
}
void State::updateBaBsrScore( const unsigned constraintIndex, const double BaBsrScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * TOTAL_FEATURES + BaBsr_SCORE] = BaBsrScore;
}


void State::updateGlobalFeatures( unsigned unstableCount, unsigned treeDepth, unsigned splitsSoFar )
{
    for ( unsigned i = 0; i < _numConstraints; ++i )
    {
        const size_t base = i * TOTAL_FEATURES;
        _stateData[base + NUM_LOCAL_FEATURES + GF_UNSTABLE_COUNT] =
            static_cast<double>( unstableCount );
        _stateData[base + NUM_LOCAL_FEATURES + GF_TREE_DEPTH] = static_cast<double>( treeDepth );
        _stateData[base + NUM_LOCAL_FEATURES + GF_SPLITS_SO_FAR] =
            static_cast<double>( splitsSoFar );
    }
}