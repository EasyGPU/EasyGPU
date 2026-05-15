/**
 * @file Store.cpp
 * @brief Implementation of store IR nodes for writing to buffers and variables.
 */

#include <IR/Node/Store.h>

namespace GPU::IR::Node {
StoreNode::StoreNode(std::unique_ptr<Node> LHS, std::unique_ptr<Node> RHS)
	: _lhs(std::move(LHS)), _rhs(std::move(RHS)) {
}

NodeType StoreNode::Type() const {
	return NodeType::Store;
}

const Node *StoreNode::LHS() const {
	return _lhs.get();
}

const Node *StoreNode::RHS() const {
	return _rhs.get();
}

std::unique_ptr<Node> StoreNode::Clone() const {
	std::unique_ptr<Node> lhsClone = _lhs ? _lhs->Clone() : nullptr;
	std::unique_ptr<Node> rhsClone = _rhs ? _rhs->Clone() : nullptr;
	return std::make_unique<StoreNode>(std::move(lhsClone), std::move(rhsClone));
}
} // namespace GPU::IR::Node
