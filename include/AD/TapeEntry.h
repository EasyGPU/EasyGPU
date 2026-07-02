#pragma once

/**
 * @file TapeEntry.h
 * @brief Core data structures for the automatic differentiation tape.
 *
 * Defines TapeVar (a named variable on the tape), TapeOpKind (categories of
 * differentiable operations), and TapeEntry (a single recorded operation).
 */

#ifndef EASYGPU_AD_TAPEENTRY_H
#define EASYGPU_AD_TAPEENTRY_H

#include <IR/Node/CompoundAssignment.h>
#include <IR/Node/Operation.h>

#include <cstdint>
#include <string>
#include <vector>

namespace GPU::AD {

/**
 * A named variable on the gradient tape.
 * Variables are identified by their GLSL name string (e.g. "v5", "buf0[v3]", "v3.xyz").
 */
struct TapeVar {
	std::string name;
	std::string glslType;
	bool		isParameter = false;
};

/**
 * Categories of differentiable operations that can be recorded on the tape.
 */
enum class TapeOpKind : uint8_t {
	BinaryOp,			// v = a + b, a * b, a / b, a - b
	UnaryOp,			// v = -a
	ExpressionGradient, // v = complex expression, with precomputed leaf gradient coefficients
	Intrinsic1,			// v = sin(x), sqrt(x), exp(x) ...  (1 parameter)
	Intrinsic2,			// v = pow(a,b), atan2(y,x) ...       (2 parameters)
	Intrinsic3,			// v = clamp(x,lo,hi), mix(a,b,t) ... (3 parameters)
	Ternary,			// v = cond ? a : b
	CompoundAssign,		// v += a, v *= a ...
	Call,				// v = callable_func(args...) — user-defined function call
	Return,				// return v — marks the return variable in a callable body
	ControlFlowBegin,	// Entering if / for block
	ControlFlowEnd,		// Leaving if / for block
	Loss,				// Marks the scalar loss variable (seed for backward pass)
};

/**
 * Distinguishes what kind of control flow a ControlFlowBegin marker represents.
 */
enum class ControlFlowKind : uint8_t {
	IfBranch,	// if(condition)
	ElifBranch, // else if(condition)
	ElseBranch, // else (no condition)
	ForLoop,	// for(start, end, step)
};

/**
 * A single entry on the gradient tape, representing one differentiable operation.
 */
struct TapeEntry {
	int32_t								  id   = 0;
	TapeOpKind							  kind = TapeOpKind::BinaryOp;
	TapeVar								  output;
	std::vector<TapeVar>				  inputs;

	// Operation-specific data
	GPU::IR::Node::OperationCode		  binaryOp	 = GPU::IR::Node::OperationCode::Add;
	GPU::IR::Node::CompoundAssignmentCode compoundOp = GPU::IR::Node::CompoundAssignmentCode::AddAssign;
	std::string							  intrinsicName;
	std::string							  callableFuncName; // for Call: the mangled GLSL function name
	int callableIndex = -1;					 // for Call: index into the sub-tape list (assigned during body recording)
	std::vector<std::string> inputGradExprs; // for ExpressionGradient: coefficient per input
	std::vector<std::string> inputGradTypes; // GLSL type for each coefficient expression
	std::string forwardExpr;				 // rematerializable forward RHS expression, used when inlining callable AD

	// Control flow metadata (only valid for ControlFlowBegin)
	ControlFlowKind			 controlFlowKind = ControlFlowKind::IfBranch;
	std::string				 conditionVarName;
	std::string				 forStart;
	std::string				 forEnd;
	std::string				 forStep = "1";
	std::string				 forVarName;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_TAPEENTRY_H
