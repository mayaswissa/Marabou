#include "DQNState.h"

State::State( unsigned numConstraints, unsigned numPhases )
    : _stateData( numConstraints, std::vector<int>( numPhases, 0 ) )
    , _numPhases( numPhases )
{
    // set all phases not fixed
    for ( unsigned i = 0; i < numConstraints; ++i )
    {
        _stateData[i][0] = 1;
    }
}

State::State(const State& other)
    : _stateData(other._stateData),
      _numPhases(other._numPhases)
{
}
State& State::operator=(const State& other) {
    if (this == &other) {
        return *this;
    }

    _stateData = other._stateData;
    _numPhases = other._numPhases;

    return *this;
}

torch::Tensor State::toTensor() const
{
    std::vector<float> flatData;
    for ( const auto &constraint : _stateData )
    {
        for ( int phase : constraint )
        {
            flatData.push_back( static_cast<float>( phase ) );
        }
    }
    auto tensor = torch::tensor(flatData, torch::kInt64);

    return tensor.view({1, static_cast<long>(flatData.size())});
}

void State::updatConstraintPhase( unsigned constraintIndex, unsigned newPhase )
{
    if ( constraintIndex < static_cast<unsigned>(_stateData.size()) && newPhase < _numPhases )
    {
        // reset this constraint's vector to zeros and assign 1 to the new phase's enrty
        std::fill( _stateData[constraintIndex].begin(), _stateData[constraintIndex].end(), 0 );
        _stateData[constraintIndex][newPhase] = 1;
    }
}

const std::vector<std::vector<int>> &State::getData() const
{
    return _stateData;
}

int State::encodeStateIndex( const std::pair<int, int> &element ) const
{
    return element.first * _numPhases + element.second;
}