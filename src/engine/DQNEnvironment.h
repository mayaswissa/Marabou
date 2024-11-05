#ifndef DQNENVIRONMENT_H
#define DQNENVIRONMENT_H
#include "List.h"
#include "NetworkLevelReasoner.h"
#include "PiecewiseLinearConstraint.h"
#include "Vector.h"

#undef Warning
#include <torch/torch.h>

// enum PhaseStatusDQN : unsigned
// {
//   PHASE_NOT_FIXED = 0,
//   RELU_PHASE_ACTIVE = 1,
//   RELU_PHASE_INACTIVE = 2,
// };

struct CustomAction
{
    unsigned int constraintIndex;
    PhaseStatus assignment;
    CustomAction( const unsigned int constraintIndex, const PhaseStatus assignment )
        : constraintIndex( constraintIndex )
        , assignment( assignment )
    {
    }
};


class Environment
{
public:
    Environment( const InputQuery &inputQuery, const ITableau &tableau );

    // Reset the environment to an initial state
    void reset( torch::Tensor &initialState );

    // Step function to advance the environment state given an action
    void step( const torch::Tensor &actionTensor,
               torch::Tensor &nextState,
               unsigned &reward,
               bool &done );
    bool isDone() const;


private:
    const List<PiecewiseLinearConstraint *> &_plConstraints;
    NLR::NetworkLevelReasoner *_networkLevelReasoner;
    unsigned _numberOfVariables;
    const ITableau &_tableau;
    /*
      The representation of the current phase pattern (one linear phase of the
      non-linear SoI function) as a mapping from PLConstraints to phase patterns.
    */
    Map<PiecewiseLinearConstraint *, PhaseStatus> _currentPhasePattern;

    /*
    The representation of the current state of the environment.
    As a mapping from neuron to its assignment (phase pattern)
     */
    Map<unsigned, PhaseStatus> _currentState;

    /*
      The constraints in the current phase pattern (i.e., participating in the
      SoI) stored in a Vector for ease of random access.
    */
    Vector<PiecewiseLinearConstraint *> _plConstraintsInCurrentPhasePattern;

    /*
      A local copy of the current variable assignment, which is refreshed via
      the obtainCurrentAssignment() method.
    */
    Map<unsigned, double> _currentAssignment;

    /*
      The constraints whose cost terms were changed in the last proposal.
      We keep track of this for the PseudoImpact branching heuristics.
    */
    List<PiecewiseLinearConstraint *> _constraintsUpdatedInLastProposal;

    /*
    Piecewise linear constraints that are currently violated.
  */
    List<PiecewiseLinearConstraint *> _violatedPlConstraints;

    torch::Device device;

    /*
      Clear _currentPhasePattern, _lastAcceptedPhasePattern and
      _plConstraintsInCurrentPhasePattern.
    */
    void resetPhasePattern();

    /*
      Set _currentPhasePattern according to the current input assignment.
    */
    void initializePhasePatternWithCurrentInputAssignment();

    /*
      Set _currentPhasePattern according to the current assignment.
    */
    void initializePhasePatternWithCurrentAssignment();

    void collectViolatedPlConstraints();
    /*
      This method computes the cost reuduction of a plConstraint participating
      in the phase pattern. The cost reduction is the largest value by which the
      cost (w.r.t. the current assignment) will decrease if we choose a
      different phase for the plConstraint in the phase pattern. This value is
      stored in reducedCost. The phase corresponding to the largest reduction
      is stored in phaseOfReducedCost.
      Note that the phase can be negative, which means the current phase is
      (locally) optimal.
    */
    void getCostReduction( PiecewiseLinearConstraint *plConstraint,
                           double &reducedCost,
                           PhaseStatus &phaseOfReducedCost ) const;

    /*
     Obtain the current variable assignment from the Tableau.
    */
    void obtainCurrentAssignment();
};

#endif
