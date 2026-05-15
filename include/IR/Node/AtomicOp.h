#pragma once

/**
 * @file AtomicOp.h
 * @brief The node for atomic operations.
 */

#ifndef EASYGPU_ATOMICOP_H
#define EASYGPU_ATOMICOP_H

#include <IR/Node/Node.h>
#include <memory>

namespace GPU::IR::Node {
/**
 * The atomic operation code
 */
enum class AtomicOpCode {
	Add,	  // atomicAdd
	Sub,	  // atomicSub (GLSL 4.6+)
	Min,	  // atomicMin
	Max,	  // atomicMax
	And,	  // atomicAnd
	Or,		  // atomicOr
	Xor,	  // atomicXor
	Exchange, // atomicExchange
	CompSwap  // atomicCompSwap
};

/**
 * The node for atomic operations
 */
class AtomicOpNode : public Node {
public:
	/**
	 * Constructor for binary atomic operations (most ops)
	 * @param code The atomic operation code
	 * @param target The target memory location (array access or variable)
	 * @param value The value to apply
	 */
	AtomicOpNode(AtomicOpCode code, std::unique_ptr<Node> target, std::unique_ptr<Node> value);

	/**
	 * Constructor for compare-and-swap (ternary op)
	 * @param target The target memory location
	 * @param compare The comparison value
	 * @param value The new value to set if comparison succeeds
	 */
	AtomicOpNode(std::unique_ptr<Node> target, std::unique_ptr<Node> compare, std::unique_ptr<Node> value);

public:
	NodeType Type() const override;

public:
	/**
	 * Getting the atomic operation code
	 * @return The atomic operation code
	 */
	[[nodiscard]] AtomicOpCode			Code() const;

	/**
	 * Getting the target memory location
	 * @return The target node (typically ArrayAccessNode or LoadLocalArray)
	 */
	[[nodiscard]] const Node		   *Target() const;

	/**
	 * Getting the value operand
	 * @return The value node
	 */
	[[nodiscard]] const Node		   *Value() const;

	/**
	 * Getting the compare value (for CompSwap only)
	 * @return The compare node, or nullptr if not CompSwap
	 */
	[[nodiscard]] const Node		   *Compare() const;

	/**
	 * Check if this is a compare-and-swap operation
	 * @return true if this is CompSwap
	 */
	[[nodiscard]] bool					IsCompSwap() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	AtomicOpCode		  _code;
	std::unique_ptr<Node> _target;
	std::unique_ptr<Node> _value;
	std::unique_ptr<Node> _compare; // Only used for CompSwap
};
} // namespace GPU::IR::Node

#endif // EASYGPU_ATOMICOP_H
