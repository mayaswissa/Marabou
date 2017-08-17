/*********************                                                        */
/*! \file MaxConstraint.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Derek Huang
 ** This file is part of the Marabou project.
 ** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "Debug.h"
#include "FloatUtils.h"
#include "FreshVariables.h"
#include "ITableau.h"
#include "MaxConstraint.h"
#include "PiecewiseLinearCaseSplit.h"
#include "ReluplexError.h"
#include <algorithm>

MaxConstraint::MaxConstraint( unsigned f, const Set<unsigned> &elements )
	: _f( f )
	, _elements( elements )
	, _minLowerBound( FloatUtils::infinity() )
	, _maxUpperBound( FloatUtils::negativeInfinity() )
	, _phaseFixed( false )
{
}

MaxConstraint::~MaxConstraint()
{
	_elements.clear();
}

PiecewiseLinearConstraint *MaxConstraint::duplicateConstraint() const
{
    MaxConstraint *clone = new MaxConstraint( _f, _elements );
	*clone = *this;
    return clone;
}

void MaxConstraint::restoreState( const PiecewiseLinearConstraint *state )
{
	const MaxConstraint *max = dynamic_cast<const MaxConstraint *>( state );
	*this = *max;
}

void MaxConstraint::registerAsWatcher( ITableau *tableau )
{
	tableau->registerToWatchVariable( this, _f );
	for ( unsigned element : _elements )
		tableau->registerToWatchVariable( this, element );
}

void MaxConstraint::unregisterAsWatcher( ITableau *tableau )
{
	tableau->unregisterToWatchVariable( this, _f );
	for ( unsigned element : _elements )
		tableau->unregisterToWatchVariable( this, element );
}

void MaxConstraint::notifyVariableValue( unsigned variable, double value )
{
	if ( variable != _f )
	{
	//Two conditions for _maxIndex to not exist: either _assignment.size()
	//equals to 0, or the only element in _assignment is _f.
	//Otherwise, we only replace _maxIndex if the value of _maxIndex is less
	//than the new value.
		if ( _assignment.size() == 0 || ( _assignment.exists( _f ) &&
	   		 _assignment.size() == 1 ) || _assignment.get( _maxIndex ) < value )
			 _maxIndex = variable;
	}
	_assignment[variable] = value;
}

double MaxConstraint::getMinLowerBound() const
{
	return (_lowerBounds.keys() == _elements) ? _minLowerBound : FloatUtils::negativeInfinity();
}

double MaxConstraint::getMaxUpperBound() const
{
	return (_upperBounds.keys() == _elements) ? _maxUpperBound : FloatUtils::infinity();
}

void MaxConstraint::notifyLowerBound( unsigned variable, double value )
{
	if ( _lowerBounds.exists( variable ) && !FloatUtils::gt( value, _lowerBounds[variable] ) )
		return;

	_lowerBounds[variable] = value;

	if ( FloatUtils::lt( value, _minLowerBound ) )
	{
		_minLowerBound = value;
		_entailedTightenings.push( Tightening( _f, _minLowerBound, Tightening::LB ) );
	}

	// If all elements except this one are bounded above and this lower bound is greater, then phase is fixed.
	if ( _elements.exists( variable ) )
	{
		double maxUpperBound = FloatUtils::negativeInfinity();
		Set<unsigned> elements = _elements;
		elements.erase( variable );
		for ( auto otherVariable : elements )
		{
			if ( _upperBounds.exists( otherVariable ) && FloatUtils::gt( _upperBounds[otherVariable], maxUpperBound ) )
			{
				maxUpperBound = _upperBounds[otherVariable];
			}
			else
			{
				maxUpperBound = FloatUtils::infinity();
				break;
			}
		}

		if ( FloatUtils::gt( value, getMaxUpperBound() ) )
		{
			_phaseFixed = true;
			_fixedPhase = variable;
		}
	}
}

void MaxConstraint::notifyUpperBound( unsigned variable, double value )
{
    if ( _upperBounds.exists( variable ) && !FloatUtils::lt( value, _upperBounds[variable] ) )
		return;

	_upperBounds[variable] = value;

	if ( FloatUtils::gt( value, _maxUpperBound ) )
	{
		_maxUpperBound = value;
		_entailedTightenings.push( Tightening( _f, _maxUpperBound, Tightening::UB ) );
	}
}

bool MaxConstraint::participatingVariable( unsigned variable ) const
{
    return ( variable == _f ) || _elements.exists( variable );
}

List<unsigned> MaxConstraint::getParticipatingVariables() const
{
	List<unsigned> temp;
	for ( auto element : _elements )
		temp.append( element );
	temp.append( _f );
	return temp;
}

bool MaxConstraint::satisfied() const
{
	if ( !( _assignment.exists( _f )  &&  _assignment.size() > 1 ) )
		throw ReluplexError( ReluplexError::PARTICIPATING_VARIABLES_ABSENT );

	double fValue = _assignment.get( _f );
	return FloatUtils::areEqual( _assignment.get( _maxIndex ), fValue );
}

List<PiecewiseLinearConstraint::Fix> MaxConstraint::getPossibleFixes() const
{
	ASSERT( !satisfied() );
	ASSERT( _assignment.exists( _f ) && _assignment.size() > 1 );

	double fValue = _assignment.get( _f );
	double maxVal = _assignment.get( _maxIndex );

	List<PiecewiseLinearConstraint::Fix> fixes;

	// Possible violations
	//	1. f is greater than maxVal
	//	2. f is less than maxVal

	if ( FloatUtils::gt( fValue, maxVal ) )
	{
		fixes.append( PiecewiseLinearConstraint::Fix( _f, maxVal ) );
		for ( auto elem : _elements )
		{
			fixes.append( PiecewiseLinearConstraint::Fix( elem, fValue ) );
		}

	}
	else if ( FloatUtils::lt( fValue, maxVal ) )
	{
		fixes.append( PiecewiseLinearConstraint::Fix( _f, maxVal ) );

		/*for ( auto elem : _elements )
		{
			if ( _assignment.exists( elem ) && FloatUtils::lt( fValue, _assignment.get( elem ) ) )
				fixes.append( PiecewiseLinearConstraint::Fix( elem, fValue ) );
		}*/
	}
	return fixes;
}

List<PiecewiseLinearCaseSplit> MaxConstraint::getCaseSplits() const
{
    ASSERT(	_assignment.exists( _f ) );

	List<PiecewiseLinearCaseSplit> splits;
	for ( unsigned element : _elements )
	{
		if ( _assignment.exists( element ) )
			splits.append( getSplit( element ) );
	}
	return splits;
}

bool MaxConstraint::phaseFixed() const
{
	return _phaseFixed;
}

PiecewiseLinearCaseSplit MaxConstraint::getValidCaseSplit() const
{
	return getSplit( _fixedPhase );
}

PiecewiseLinearCaseSplit MaxConstraint::getSplit( unsigned argMax ) const
{
	ASSERT( _assignment.exists( argMax ) );
	PiecewiseLinearCaseSplit maxPhase;

	// maxArg - f = 0
	Equation maxEquation;
	maxEquation.addAddend( 1, argMax );
	maxEquation.addAddend( -1, _f );
	maxEquation.setScalar( 0 );
	maxPhase.addEquation( maxEquation, PiecewiseLinearCaseSplit::EQ );

	for ( unsigned other : _elements )
	{
		if ( argMax == other )
			continue;

		Equation gtEquation;

		// argMax >= other
		gtEquation.addAddend( -1, other );
		gtEquation.addAddend( 1, argMax );
		gtEquation.setScalar( 0 );
		maxPhase.addEquation( gtEquation, PiecewiseLinearCaseSplit::GE );
	}
	
	return maxPhase;
}

void MaxConstraint::updateVarIndex( unsigned prevVar, unsigned newVar )
{
	ASSERT( participatingVariable( prevVar ) );

	if ( _assignment.exists( prevVar ) )
	{
		_assignment[newVar] = _assignment.get( prevVar );
		_lowerBounds[newVar] = _lowerBounds.get( prevVar  );
		_upperBounds[newVar] = _upperBounds.get( prevVar );
		_assignment.erase( prevVar );
	}

	if ( prevVar == _maxIndex )
		_maxIndex = newVar;

	if ( prevVar == _f )
		_f = newVar;
	else
	{
		_elements.erase( prevVar );
		_elements.insert( newVar );
	}
}

void MaxConstraint::eliminateVar( unsigned, double )
{
}

//
// Local Variables:
// compile-command: "make -C .. "
// tags-file-name: "../TAGS"
// c-basic-offset: 4
// End:
//