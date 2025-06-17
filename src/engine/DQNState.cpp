#include "DQNState.h"

State::State( const unsigned numConstraints )
    : _stateData( numConstraints )
    , _numConstraints( numConstraints )
    , _numPhases( DQN_NUM_PHASES )
{
    // Allocate a contiguous vector with (numConstraints * NUM_FEATURES) elements.
    _stateData.resize( numConstraints * NUM_FEATURES, 0.0 );

    // For each constraint, set the feature at index DQN_RELU_NOT_FIXED to 1.0,
    // SoI scores to 0.
    for ( unsigned i = 0; i < numConstraints; ++i )
    {
        _stateData[i * NUM_FEATURES + DQN_RELU_NOT_FIXED] = 1.0;
        // _stateData[i * NUM_FEATURES + SOI_ACTIVE_SCORE] = 0;
        // _stateData[i * NUM_FEATURES + SOI_INACTIVE_SCORE] = 0;
        _stateData[i * NUM_FEATURES + BaBsr_SCORE] = 0;
    }
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
    auto stateTensor = torch::tensor( _stateData );
    stateTensor = stateTensor.to( torch::kFloat32 );
    return stateTensor.view(
        { static_cast<long>( _numConstraints ), static_cast<long>( NUM_FEATURES ) } );
}


void State::updateConstraintPhase( const unsigned constraintIndex, const unsigned newPhase )
{
    if ( constraintIndex >= _numConstraints || newPhase >= _numPhases )
        return;
    // Get pointer to the start of the row for this constraint.
    double *rowPtr = &_stateData[constraintIndex * NUM_FEATURES];
    // Reset the first _numPhases entries (phase indicators) to 0.0.
    std::fill_n( rowPtr, _numPhases, 0.0 );
    // Set the new phase.
    rowPtr[newPhase] = 1.0;
}
// void State::updateSoIScoreForAgent( const unsigned constraintIndex,
//                             const double SoiActiveScore,
//                             const double SoiInactiveScore )
// {
//     if ( constraintIndex >= _numConstraints )
//         return;
//     _stateData[constraintIndex * NUM_FEATURES + SOI_ACTIVE_SCORE] = SoiActiveScore;
//     _stateData[constraintIndex * NUM_FEATURES + SOI_INACTIVE_SCORE] = SoiInactiveScore;
// }

void State::updatePolarity( const unsigned constraintIndex, const double polarityScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * NUM_FEATURES + POLARITY_SCORE] = polarityScore;
}

void State::updateBaBsrScore( const unsigned constraintIndex, const double BaBsrScore )
{
    if ( constraintIndex >= _numConstraints )
        return;
    _stateData[constraintIndex * NUM_FEATURES + BaBsr_SCORE] = BaBsrScore;
}
