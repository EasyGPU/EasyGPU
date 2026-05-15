/**
 * @file Continue.cpp
 * @brief Implementation of the continue statement IR node.
 */

#include <IR/Node/Continue.h>

namespace GPU::IR::Node {
NodeType ContinueNode::Type() const {
	return NodeType::Continue;
}

std::unique_ptr<Node> ContinueNode::Clone() const {
	return std::make_unique<ContinueNode>();
}
} // namespace GPU::IR::Node
