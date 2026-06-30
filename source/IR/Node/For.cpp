/**
 * @file For.cpp
 * @brief Implementation of the for loop IR node.
 */

#include <IR/Node/For.h>

namespace GPU::IR::Node {
ForNode::ForNode(const std::string &VarName, int Start, int End, int Step, std::vector<std::unique_ptr<Node>> &Body)
	: _varName(VarName), _start(Start), _end(End), _step(Step), _body(std::move(Body)) {
}

ForNode::ForNode(std::vector<std::unique_ptr<Node>> &Init, std::unique_ptr<Node> &Condition,
				 std::vector<std::unique_ptr<Node>> &Step, std::vector<std::unique_ptr<Node>> &Body)
	: _start(0), _end(0), _step(0), _hasDynamicHeader(true), _init(std::move(Init)),
	  _condition(std::move(Condition)), _stepNodes(std::move(Step)), _body(std::move(Body)) {
}

NodeType ForNode::Type() const {
	return NodeType::For;
}

const std::string &ForNode::VarName() const {
	return _varName;
}

int ForNode::Start() const {
	return _start;
}

int ForNode::End() const {
	return _end;
}

int ForNode::Step() const {
	return _step;
}

bool ForNode::HasDynamicHeader() const {
	return _hasDynamicHeader;
}

const std::vector<std::unique_ptr<Node>> &ForNode::Init() const {
	return _init;
}

const std::unique_ptr<Node> &ForNode::Condition() const {
	return _condition;
}

const std::vector<std::unique_ptr<Node>> &ForNode::StepNodes() const {
	return _stepNodes;
}

const std::vector<std::unique_ptr<Node>> &ForNode::Body() const {
	return _body;
}

std::unique_ptr<Node> ForNode::Clone() const {
	std::vector<std::unique_ptr<Node>> bodyClone;
	bodyClone.reserve(_body.size());
	for (const auto &node : _body) {
		if (node) {
			bodyClone.push_back(node->Clone());
		} else {
			bodyClone.push_back(nullptr);
		}
	}

	if (!_hasDynamicHeader) {
		return std::make_unique<ForNode>(_varName, _start, _end, _step, bodyClone);
	}

	std::vector<std::unique_ptr<Node>> initClone;
	initClone.reserve(_init.size());
	for (const auto &node : _init) {
		if (node) {
			initClone.push_back(node->Clone());
		} else {
			initClone.push_back(nullptr);
		}
	}

	std::unique_ptr<Node> conditionClone = _condition ? _condition->Clone() : nullptr;

	std::vector<std::unique_ptr<Node>> stepClone;
	stepClone.reserve(_stepNodes.size());
	for (const auto &node : _stepNodes) {
		if (node) {
			stepClone.push_back(node->Clone());
		} else {
			stepClone.push_back(nullptr);
		}
	}

	return std::make_unique<ForNode>(initClone, conditionClone, stepClone, bodyClone);
}
} // namespace GPU::IR::Node
