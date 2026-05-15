/**
 * @file LoadUniform.cpp
 * @brief Implementation of uniform load IR node.
 */

#include <IR/Node/LoadUniform.h>

namespace GPU::IR::Node {
LoadUniformNode::LoadUniformNode(std::string Uniform) : _uniform(std::move(Uniform)) {
}

std::string LoadUniformNode::Unwrap() const {
	return _uniform;
}

std::unique_ptr<Node> LoadUniformNode::Clone() const {
	return std::make_unique<LoadUniformNode>(_uniform);
}
} // namespace GPU::IR::Node
