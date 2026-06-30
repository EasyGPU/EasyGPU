/**
 * @file LocalVariable.cpp
 * @brief Implementation of local variable declaration IR node.
 */

#include <IR/Node/LocalVariable.h>

#include <utility>

namespace GPU::IR::Node {
LocalVariableNode::LocalVariableNode(std::string Name, std::string Type)
	: _name(std::move(Name)), _type(std::move(Type)), _isExternal(false) {
}

LocalVariableNode::LocalVariableNode(std::string Name, std::string Type, std::unique_ptr<Node> Initializer)
	: _name(std::move(Name)), _type(std::move(Type)), _isExternal(false), _initializer(std::move(Initializer)) {
}

LocalVariableNode::LocalVariableNode(std::string Name, std::string Type, bool IsExternal)
	: _name(std::move(Name)), _type(std::move(Type)), _isExternal(IsExternal) {
}

std::string LocalVariableNode::VarName() const {
	return _name;
}

std::string LocalVariableNode::VarType() const {
	return _type;
}

bool LocalVariableNode::IsExternal() const {
	return _isExternal;
}

bool LocalVariableNode::HasInitializer() const {
	return _initializer != nullptr;
}

const Node *LocalVariableNode::Initializer() const {
	return _initializer.get();
}

NodeType LocalVariableNode::Type() const {
	return NodeType::LocalVariable;
}

std::unique_ptr<Node> LocalVariableNode::Clone() const {
	auto clone = _initializer == nullptr
		? std::make_unique<LocalVariableNode>(_name, _type, _isExternal)
		: std::make_unique<LocalVariableNode>(_name, _type, _initializer->Clone());
	return clone;
}
} // namespace GPU::IR::Node
