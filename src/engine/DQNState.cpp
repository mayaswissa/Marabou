#include "DQNState.h"

#include <GlobalConfiguration.h>

State::State( const unsigned numConstraints, const unsigned numPhases )
    : _stateData( numConstraints, std::vector<double>( numPhases + 2, 0.0f ) )
    , _numPhases( numPhases )
{
    // set all phases not fixed
    for ( unsigned i = 0; i < numConstraints; ++i )
    {
        _stateData[i][0] = 1.0;              // Default not fixed phase
    }
}

State::State( const State &other )
    : _stateData( other._stateData )
    , _numPhases( other._numPhases )
{
}
State &State::operator=( const State &other )
{
    if ( this == &other )
        return *this;

    _stateData = other._stateData;
    _numPhases = other._numPhases;
    return *this;
}

torch::Tensor State::toTensor() const {
    std::vector<int64_t> phaseData;
    std::vector< double> boundsData;
    unsigned numConstraints = _stateData.size();
    unsigned counter = 0;
    for (const auto& constraint : _stateData) {
        counter += constraint.size();
        for (size_t i = 0; i < constraint.size() - 2; ++i)
            phaseData.push_back(static_cast<int64_t>(constraint[i]));  // Collect phase indices
        double upperBound = std::tanh(constraint[constraint.size() - 2]);
        double lowerBound = std::tanh(constraint[constraint.size() - 1]);
        boundsData.push_back(upperBound);
        boundsData.push_back(lowerBound);
    }
    auto phaseTensor = torch::tensor(phaseData, torch::kInt64).view({numConstraints, _numPhases});
    auto boundsTensor = torch::tensor(boundsData, torch::kFloat32).view({numConstraints, 2});
    auto tensorState = torch::cat({phaseTensor, boundsTensor}, 1);
    return tensorState;
}


void State::updateConstraintPhase( const unsigned constraintIndex, const unsigned newPhase )
{
    if ( constraintIndex < static_cast<unsigned>( _stateData.size() ) && newPhase < _numPhases )
    {
        // reset this constraint's vector to zeros and assign 1 to the new phase's entry
        std::fill_n( _stateData[constraintIndex].begin(),
                   _numPhases,
                   0.0f );
        _stateData[constraintIndex][newPhase] = 1.0f;
    }
}

void State::updateBounds( const unsigned constraintIndex,
                       const double upperBound,
                       const double lowerBound )
{
    if ( constraintIndex < _stateData.size() )
    {
        _stateData[constraintIndex][_numPhases] = upperBound;
        _stateData[constraintIndex][_numPhases + 1] = lowerBound;
    }
}

const std::vector<std::vector<double>> &State::getData() const
{
    return _stateData;
}

bool State::constraintActive(const unsigned constraintIndex) const
{
    return this->getData()[constraintIndex][GlobalConfiguration::DQN_CONSTRAINT_INACTIVE] == 0;
}