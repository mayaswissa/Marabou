#include "DQNEnvironment.h"
#include "Options.h"
#include <InputQuery.h>

torch::Tensor customActionToTensor(const CustomAction& action) {
    return torch::tensor({static_cast<int>(action.constraintIndex), static_cast<int>(action.assignment)},
                                 torch::dtype(torch::kInt32));
}

CustomAction tensorToCustomAction(const torch::Tensor& tensor) {
    const unsigned int constraintIndex = tensor[0].item<int>();
    const auto assignment = static_cast<PhaseStatus>(tensor[1].item<int>());
    return CustomAction(constraintIndex, assignment);
}

Environment::Environment( const InputQuery &inputQuery,const ITableau &tableau )
    : _plConstraints( inputQuery.getPiecewiseLinearConstraints() )
    , _networkLevelReasoner( inputQuery.getNetworkLevelReasoner() )
    , _numberOfVariables( inputQuery.getNumberOfVariables() )
    , _tableau( tableau )
    ,device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    initializePhasePatternWithCurrentInputAssignment();
    initializePhasePatternWithCurrentAssignment();
    collectViolatedPlConstraints();
}

void Environment::reset(torch::Tensor& initialState)
{
    _currentPhasePattern.clear();
    _plConstraintsInCurrentPhasePattern.clear();
    _constraintsUpdatedInLastProposal.clear();
    initializePhasePatternWithCurrentInputAssignment();
    convertPhasePatternToState( _currentPhasePattern, initialState );
}

void Environment::initializePhasePatternWithCurrentAssignment()
{
    obtainCurrentAssignment();

    for ( const auto &plConstraint : _plConstraints )
    {
        ASSERT( !_currentPhasePattern.exists( plConstraint ) );
        if ( plConstraint->isActive() && !plConstraint->phaseFixed() )
        {
            // Set the phase status corresponding to the current assignment.
            _currentPhasePattern[plConstraint] =
                plConstraint->getPhaseStatusInAssignment( _currentAssignment );
        }
    }
}

void Environment::initializePhasePatternWithCurrentInputAssignment()
{
    ASSERT( _networkLevelReasoner );
    Map<unsigned, double> assignment;
    _networkLevelReasoner->concretizeInputAssignment( assignment );

    for ( const auto &plConstraint : _plConstraints )
    {
        ASSERT( !_currentPhasePattern.exists( plConstraint ) );
        if ( plConstraint->supportSoI() && plConstraint->isActive() && !plConstraint->phaseFixed() )
        {
            // Set the phase status corresponding to the current assignment.
            _currentPhasePattern[plConstraint] =
                plConstraint->getPhaseStatusInAssignment( assignment );
        }
    }
}

void Environment::convertPhasePatternToState(Map<PiecewiseLinearConstraint *, PhaseStatus>& phasePattern, torch::Tensor &state)
{
    _currentState.clear();
    for (const auto &pair : phasePattern )
    {
        PhaseStatus phaseStatus = pair.second;
        unsigned phaseIndex = phaseStatusToIndex[phaseStatus];
        _currentState.push_back( phaseIndex );
        state = torch::tensor( &_currentState, torch::dtype(torch::kInt64));
    }
}


void Environment::obtainCurrentAssignment()
{
    _currentAssignment.clear();
    for ( unsigned i = 0; i < _numberOfVariables; ++i )
        _currentAssignment[i] = _tableau.getValue( i );
}

void Environment::step( const torch::Tensor& actionTensor, torch::Tensor& nextState, unsigned& reward, bool& done) {

    _constraintsUpdatedInLastProposal.clear();
    const auto action = tensorToCustomAction( actionTensor );
    PiecewiseLinearConstraint *plConstraintToUpdate =
        _plConstraintsInCurrentPhasePattern[action.constraintIndex];
    _currentPhasePattern[plConstraintToUpdate] = action.assignment;
    _constraintsUpdatedInLastProposal.append( plConstraintToUpdate );

    // todo check this part:
    obtainCurrentAssignment();
    for ( const auto &pair : _currentPhasePattern )
    {
        if ( pair.first->satisfied() )
        {
            PhaseStatus satisfiedPhaseStatus =
                pair.first->getPhaseStatusInAssignment( _currentAssignment );
            _currentPhasePattern[pair.first] = satisfiedPhaseStatus;
        }
    }

    // todo up to this part

    // todo check how to update the state after changing one neuron - line 300 engine

    // calculate reward - minus sum of violated constraints after this action
    collectViolatedPlConstraints();
    if (!_violatedPlConstraints.exists( plConstraintToUpdate ))
        reward = -_numberOfVariables; // todo check - action did not change the state
    else
        reward = - _violatedPlConstraints.size();
    done = isDone();


    // todo check getCostReduction - to update all other constraints after action
    // next state = current state
    convertPhasePatternToState( _currentPhasePattern, nextState );
    _constraintsUpdatedInLastProposal.clear();

}
bool Environment::isDone() const
{
    return _violatedPlConstraints.empty();
}
void Environment::collectViolatedPlConstraints()
{
    _violatedPlConstraints.clear();
    for ( const auto &constraint : _plConstraints )
    {
        if ( constraint->isActive() && !constraint->satisfied() )
            _violatedPlConstraints.append( constraint );
    }
}

