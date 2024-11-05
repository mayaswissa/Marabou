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
    initialState = torch::tensor( _currentPhasePattern );
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
    /*
      First, obtain the variable assignment from the network level reasoner.
      We should be able to get the assignment of all variables participating
      in pl constraints because otherwise the NLR would not have been
      successfully constructed.
    */
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

    collectViolatedPlConstraints();
    if (!_violatedPlConstraints.exists( plConstraintToUpdate ))
        reward = -_numberOfVariables; // todo check
    else
        reward = _violatedPlConstraints.size();
    done = isDone();
    // todo check how to update the state after changing one neuron
    // todo check if need to use _currentPhasePattern or current_state

    // todo check getCostReduction - to update all other constraints after action


    nextState = torch::tensor(&_currentPhasePattern);
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

