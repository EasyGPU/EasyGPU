/**
 * Ternary.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/8/2026
 */

#include <IR/Node/Ternary.h>

namespace GPU::IR::Node {
TernaryNode::TernaryNode(std::unique_ptr<Node> condition, std::unique_ptr<Node> trueExpr, std::unique_ptr<Node> falseExpr)
	: _condition(std::move(condition)), _trueExpr(std::move(trueExpr)), _falseExpr(std::move(falseExpr)) {
}

NodeType TernaryNode::Type() const {
	return NodeType::Ternary;
}

const Node *TernaryNode::Condition() const {
	return _condition.get();
}

const Node *TernaryNode::TrueExpr() const {
	return _trueExpr.get();
}

const Node *TernaryNode::FalseExpr() const {
	return _falseExpr.get();
}

std::unique_ptr<Node> TernaryNode::Clone() const {
	std::unique_ptr<Node> conditionClone = _condition ? _condition->Clone() : nullptr;
	std::unique_ptr<Node> trueExprClone  = _trueExpr ? _trueExpr->Clone() : nullptr;
	std::unique_ptr<Node> falseExprClone = _falseExpr ? _falseExpr->Clone() : nullptr;
	return std::make_unique<TernaryNode>(std::move(conditionClone), std::move(trueExprClone), std::move(falseExprClone));
}
} // namespace GPU::IR::Node
