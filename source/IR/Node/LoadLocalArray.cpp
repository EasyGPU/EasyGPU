/**
 * @file LoadLocalArray.cpp
 * @brief Implementation of local array load IR node.
 */

#include <IR/Node/LoadLocalArray.h>

#include <format>
#include <utility>

namespace GPU::IR::Node {
LoadLocalArrayNode::LoadLocalArrayNode(std::string Name) : _name(std::move(Name)) {
}

std::string LoadLocalArrayNode::Unwrap() const {
	return _name;
}

std::unique_ptr<Node> LoadLocalArrayNode::Clone() const {
	return std::make_unique<LoadLocalArrayNode>(_name);
}
} // namespace GPU::IR::Node
