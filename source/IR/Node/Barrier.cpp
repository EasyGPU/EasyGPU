/**
 * @file Barrier.cpp
 * @brief Implementation of shader synchronization barrier IR node.
 */

#include <IR/Node/Barrier.h>

namespace GPU::IR::Node {

BarrierNode::BarrierNode(BarrierCode code) : _code(code) {
}

NodeType BarrierNode::Type() const {
	return NodeType::Barrier;
}

BarrierCode BarrierNode::Code() const {
	return _code;
}

std::unique_ptr<Node> BarrierNode::Clone() const {
	return std::make_unique<BarrierNode>(_code);
}

} // namespace GPU::IR::Node
