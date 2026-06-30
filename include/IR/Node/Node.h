#pragma once

/**
 * @file Node.h
 * @brief The base class for all the node type.
 */

#ifndef EASYGPU_NODE_H
#define EASYGPU_NODE_H

#include <memory>

namespace GPU::IR::Node {
/**
 * The type of nodes
 */
enum class NodeType {
	LocalVariable,
	LocalArray,
	SharedMemory,
	AtomicOp,
	Barrier,
	Load,
	CallInst,
	Operation,
	Store,
	ArrayAccess,
	CompoundAssignment,
	Increment,
	MemberAccess,
	If,
	While,
	DoWhile,
	For,
	TextureLoad,
	TextureStore,
	TextureSample,
	RawCode,
	Break,
	Continue,
	Return,
	Call,
	Ternary
};

/**
 * The base class for all the nodes in the IR
 */
class Node {
public:
	Node()			= default;

	virtual ~Node() = default;

public:
	/**
	 * Getting the type of the node
	 * @return The type of this node
	 */
	[[nodiscard]] virtual NodeType				Type() const  = 0;

	/**
	 * Clone the node and its children
	 * @return A deep copy of this node
	 */
	[[nodiscard]] virtual std::unique_ptr<Node> Clone() const = 0;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_NODE_H
