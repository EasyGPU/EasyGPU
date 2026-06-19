/**
 * @file CompoundAssignment.cpp
 * @brief Implementation of compound assignment IR nodes.
 */

#include <IR/Node/CompoundAssignment.h>

namespace GPU::IR::Node {
CompoundAssignmentNode::CompoundAssignmentNode(CompoundAssignmentCode code, std::unique_ptr<Node> lhs,
											   std::unique_ptr<Node> rhs)
	: _code(code), _lhs(std::move(lhs)), _rhs(std::move(rhs)) {
}

NodeType CompoundAssignmentNode::Type() const {
	return NodeType::CompoundAssignment;
}

CompoundAssignmentCode CompoundAssignmentNode::Code() const {
	return _code;
}

const Node *CompoundAssignmentNode::LHS() const {
	return _lhs.get();
}

const Node *CompoundAssignmentNode::RHS() const {
	return _rhs.get();
}

std::unique_ptr<Node> CompoundAssignmentNode::Clone() const {
	auto lhsClone = _lhs ? _lhs->Clone() : nullptr;
	auto rhsClone = _rhs ? _rhs->Clone() : nullptr;
	return std::make_unique<CompoundAssignmentNode>(_code, std::move(lhsClone), std::move(rhsClone));
}
} // namespace GPU::IR::Node
