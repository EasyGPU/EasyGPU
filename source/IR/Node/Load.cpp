/**
 * @file Load.cpp
 * @brief Implementation of load IR nodes for reading from buffers and variables.
 */

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
NodeType LoadNode::Type() const {
	return NodeType::Load;
}

// Clone() is pure virtual - implemented by derived classes
} // namespace GPU::IR::Node
