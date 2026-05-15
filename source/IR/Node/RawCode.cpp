/**
 * @file RawCode.cpp
 * @brief Implementation of raw GLSL code injection IR node.
 */

#include <IR/Node/RawCode.h>

namespace GPU::IR::Node {
RawCodeNode::RawCodeNode(std::string Code) : _code(std::move(Code)) {
}

NodeType RawCodeNode::Type() const {
	return NodeType::RawCode;
}

const std::string &RawCodeNode::Code() const {
	return _code;
}

std::unique_ptr<Node> RawCodeNode::Clone() const {
	return std::make_unique<RawCodeNode>(_code);
}
} // namespace GPU::IR::Node
