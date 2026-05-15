/**
 * @file LoadLocalVariable.cpp
 * @brief Implementation of local variable load IR node.
 */

#include <IR/Node/LoadLocalVariable.h>

#include <utility>

namespace GPU::IR::Node {
LoadLocalVariableNode::LoadLocalVariableNode(std::string Name) : _name(std::move(Name)) {
}

std::string LoadLocalVariableNode::Unwrap() const {
	return _name;
}

std::unique_ptr<Node> LoadLocalVariableNode::Clone() const {
	return std::make_unique<LoadLocalVariableNode>(_name);
}
} // namespace GPU::IR::Node
