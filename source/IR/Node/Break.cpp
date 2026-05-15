/**
 * @file Break.cpp
 * @brief Implementation of the break statement IR node.
 */

#include <IR/Node/Break.h>

namespace GPU::IR::Node {
NodeType BreakNode::Type() const {
	return NodeType::Break;
}

std::unique_ptr<Node> BreakNode::Clone() const {
	return std::make_unique<BreakNode>();
}
} // namespace GPU::IR::Node
