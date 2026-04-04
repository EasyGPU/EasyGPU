/**
 * AtomicOp.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */

#include <IR/Node/AtomicOp.h>

namespace GPU::IR::Node {
AtomicOpNode::AtomicOpNode(AtomicOpCode code, std::unique_ptr<Node> target, std::unique_ptr<Node> value)
	: _code(code), _target(std::move(target)), _value(std::move(value)), _compare(nullptr) {
}

AtomicOpNode::AtomicOpNode(std::unique_ptr<Node> target, std::unique_ptr<Node> compare, std::unique_ptr<Node> value)
	: _code(AtomicOpCode::CompSwap), _target(std::move(target)), _compare(std::move(compare)), _value(std::move(value)) {
}

NodeType AtomicOpNode::Type() const {
	return NodeType::AtomicOp;
}

AtomicOpCode AtomicOpNode::Code() const {
	return _code;
}

const Node *AtomicOpNode::Target() const {
	return _target.get();
}

const Node *AtomicOpNode::Value() const {
	return _value.get();
}

const Node *AtomicOpNode::Compare() const {
	return _compare.get();
}

bool AtomicOpNode::IsCompSwap() const {
	return _code == AtomicOpCode::CompSwap;
}

std::unique_ptr<Node> AtomicOpNode::Clone() const {
	if (IsCompSwap()) {
		return std::make_unique<AtomicOpNode>(_target->Clone(), _compare->Clone(), _value->Clone());
	}
	return std::make_unique<AtomicOpNode>(_code, _target->Clone(), _value->Clone());
}
} // namespace GPU::IR::Node
