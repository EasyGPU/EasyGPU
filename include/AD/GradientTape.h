#pragma once

/**
 * @file GradientTape.h
 * @brief The gradient tape (Wengert list) for recording forward-pass operations.
 *
 * The GradientTape is the core recording mechanism. During the forward pass,
 * each differentiable operation (arithmetic, intrinsic calls, assignments) is
 * recorded as a TapeEntry. After the forward pass, the tape is walked in
 * reverse to generate adjoint (gradient) code.
 */

#ifndef EASYGPU_AD_GRADIENTTAPE_H
#define EASYGPU_AD_GRADIENTTAPE_H

#include <AD/TapeEntry.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace GPU::IR::Node {
class Node;
class StoreNode;
class OperationNode;
class IntrinsicCallNode;
class CompoundAssignmentNode;
class TernaryNode;
class LocalVariableNode;
class LoadNode;
class CallNode;
class ReturnNode;
} // namespace GPU::IR::Node

namespace GPU::AD {

/**
 * The gradient tape records every differentiable operation during the forward pass.
 *
 * It is activated by setting it on the Builder via Builder::SetGradientTape().
 * When active, Builder::Build() calls GradientTape::Record() after normal GLSL
 * emission, storing the operation metadata for later backward-pass generation.
 *
 * The tape also tracks which variables are "active" (participate in gradient
 * flow) and manages parameter registration for gradient buffer allocation.
 */
class GradientTape {
public:
	GradientTape() = default;

	// ---- Recording -------------------------------------------------------

	/**
	 * Record a node from the Builder. Called automatically by Builder::Build()
	 * when this tape is active.
	 * @param node The IR node being built
	 * @param isStatement Whether the node is a statement (vs expression sub-term)
	 */
	void						  Record(const GPU::IR::Node::Node &node, bool isStatement);

	/** Directly add a pre-built entry to the tape (used for name remapping). */
	void						  RecordRemapped(const TapeEntry &entry);

	// ---- Parameter management --------------------------------------------

	/**
	 * Register a variable as a differentiable parameter.
	 * Parameters receive gradient buffers and their adjoints are written back
	 * to GPU buffers after the backward pass.
	 */
	void						  RegisterParameter(const std::string &name, const std::string &glslType);

	/** Check if a variable name corresponds to a registered parameter. */
	bool						  IsParameter(const std::string &name) const;

	// ---- Loss marking ----------------------------------------------------

	/** Mark the scalar loss variable. Its adjoint is initialized to 1.0. */
	void						  MarkLoss(const std::string &name, const std::string &glslType);

	/** Get the loss variable, if one was marked. */
	const std::optional<TapeVar> &LossVar() const {
		return _lossVar;
	}

	// ---- Control flow markers ----------------------------------------------

	/** Begin an if branch with the given condition expression. */
	void		BeginIfBranch(const std::string &conditionExpr);

	/** Begin an elif branch with the given condition expression. */
	void		BeginElifBranch(const std::string &conditionExpr);

	/** Begin an else branch (no condition). */
	void		BeginElseBranch();

	/** End an if/elif/else chain. */
	void		EndIfChain();

	/** Begin a for loop body. */
	void		BeginForLoop(const std::string &varName, const std::string &start, const std::string &end,
							 const std::string &step);

	/** End a for loop body. */
	void		EndForLoop();

	/** Check if the tape is currently active (has been set on the Builder). */
	static bool IsActive();

	// ---- Sub-tape support (for Callable body recording) --------------------

	/** Push a new sub-tape onto the stack. All subsequent Record() calls
	 *  go to this sub-tape until PopSubTape() is called. */
	void		PushSubTape();

	/** Pop the current sub-tape and store it. Returns the index of the stored sub-tape. */
	int			PopSubTape();

	/** Deep-copy all sub-tapes from another tape (recursively).
	 *  Used for name-remapped tape copies during nested callable adjoint
	 *  generation to preserve sub-sub-tape structure. */
	void		CloneSubTapesFrom(const GradientTape &src);

	/** Get the number of recorded sub-tapes. */
	size_t		SubTapeCount() const {
		return _subTapes.size();
	}

	/** Get a sub-tape by index. */
	const GradientTape &SubTape(int index) const {
		return *_subTapes[index];
	}

	/** Get all sub-tapes. */
	const auto &SubTapes() const {
		return _subTapes;
	}

	// ---- Access ----------------------------------------------------------

	size_t Size() const {
		return _entries.size();
	}
	const TapeEntry &operator[](int32_t i) const {
		return _entries[i];
	}
	const auto &Entries() const {
		return _entries;
	}

	/** Check if a variable is on the active path (needs gradient). */
	bool IsActive(const std::string &name) const {
		return _activeNames.count(name) > 0;
	}

	/** Get the GLSL type for a tracked variable name. */
	const std::string *GetVarType(const std::string &name) const;

	/** Get all registered parameter names and their types. */
	const auto		  &Parameters() const {
		return _paramList;
	}

	/** Get the number of registered parameters. */
	size_t ParameterCount() const {
		return _paramList.size();
	}

private:
	// ---- Internal: node analysis -----------------------------------------

	void				   RecordDirect(const GPU::IR::Node::Node &node, bool isStatement);

	void				   RecordStore(const GPU::IR::Node::StoreNode &store);
	void				   RecordCompoundAssignment(const GPU::IR::Node::CompoundAssignmentNode &node);
	void				   RecordOperation(const GPU::IR::Node::OperationNode &node, const TapeVar &output);
	void				   RecordIntrinsic(const GPU::IR::Node::IntrinsicCallNode &node, const TapeVar &output);
	void				   RecordTernary(const GPU::IR::Node::TernaryNode &node, const TapeVar &output);
	void				   RecordLocalVariable(const GPU::IR::Node::LocalVariableNode &node);
	void				   RecordCall(const GPU::IR::Node::CallNode &node, const TapeVar &output);
	void				   RecordReturn(const GPU::IR::Node::ReturnNode &node);

	/** Extract a variable name from a Load node (returns Unwrap()). */
	static std::string	   ExtractVarName(const GPU::IR::Node::Node &loadNode);

	/** Extract optional variable name if the node is a LoadLocalVariable. */
	static std::string	   TryExtractVarName(const GPU::IR::Node::Node &node);

	/** Determine the TapeOpKind from an OperationCode. */
	static TapeOpKind	   ClassifyOp(GPU::IR::Node::OperationCode code);

	/** Propagate active status: if any input is active, mark output active. */
	void				   PropagateActive(const TapeVar &output, const std::vector<TapeVar> &inputs);

	// ---- Data ------------------------------------------------------------

	std::vector<TapeEntry> _entries;
	int32_t				   _nextId = 0;

	// Maps variable name -> GLSL type (populated from LocalVariableNode declarations)
	std::unordered_map<std::string, std::string>	 _varTypes;

	// Variables whose gradient is needed (transitively reachable from loss/parameters)
	std::unordered_set<std::string>					 _activeNames;

	// Registered parameters (name -> GLSL type)
	std::unordered_map<std::string, std::string>	 _parameters;

	// Ordered parameter list preserving registration order
	std::vector<std::pair<std::string, std::string>> _paramList;

	// The scalar loss variable
	std::optional<TapeVar>							 _lossVar;

	// Sub-tape support for recording Callable function bodies
	std::vector<std::unique_ptr<GradientTape>>		 _subTapes;
	std::stack<GradientTape *>						 _subTapeStack;
	GradientTape									*_currentSubTape = nullptr;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_GRADIENTTAPE_H
