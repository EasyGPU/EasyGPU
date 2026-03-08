#pragma once

/**
 * Ternary.h:
 *      @Descripiton    :   The node for ternary conditional expression (condition ? trueExpr : falseExpr)
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/8/2026
 */
#ifndef EASYGPU_TERNARY_H
#define EASYGPU_TERNARY_H

#include <IR/Node/Node.h>

#include <memory>

namespace GPU::IR::Node {
/**
 * The node for ternary conditional expression
 * Structure: (condition) ? (trueExpr) : (falseExpr)
 */
class TernaryNode : public Node {
public:
	/**
	 * Constructor for ternary node
	 * @param condition The condition expression
	 * @param trueExpr The expression evaluated when condition is true
	 * @param falseExpr The expression evaluated when condition is false
	 */
	TernaryNode(std::unique_ptr<Node> condition, std::unique_ptr<Node> trueExpr, std::unique_ptr<Node> falseExpr);

public:
	[[nodiscard]] NodeType Type() const override;

public:
	/**
	 * Getting the condition node
	 * @return The condition node
	 */
	[[nodiscard]] const Node *Condition() const;

	/**
	 * Getting the true expression node
	 * @return The true expression node
	 */
	[[nodiscard]] const Node *TrueExpr() const;

	/**
	 * Getting the false expression node
	 * @return The false expression node
	 */
	[[nodiscard]] const Node *FalseExpr() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::unique_ptr<Node> _condition;
	std::unique_ptr<Node> _trueExpr;
	std::unique_ptr<Node> _falseExpr;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_TERNARY_H
