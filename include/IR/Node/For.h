#pragma once

/**
 * @file For.h
 * @brief The node for for loop control flow.
 */

#ifndef EASYGPU_FOR_H
#define EASYGPU_FOR_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>
#include <vector>

namespace GPU::IR::Node {
/**
 * The node for for loop control flow.
 *
 * The original EasyGPU DSL constructor models canonical counted loops. The
 * module-lowering constructor carries already-lowered header/body nodes so
 * frontend IR can preserve dynamic bounds, typed locals, and structured body
 * statements without injecting a rendered GLSL header.
 */
class ForNode : public Node {
public:
	/**
	 * Constructor for for node
	 * @param VarName The loop variable name
	 * @param Start The loop start value (inclusive)
	 * @param End The loop end value (exclusive)
	 * @param Step The loop step value
	 * @param Body The loop body statements
	 */
	ForNode(const std::string &VarName, int Start, int End, int Step, std::vector<std::unique_ptr<Node>> &Body);

	/**
	 * Constructor for a dynamically-shaped for node.
	 * @param Init The initializer statement expressions for the loop header.
	 * @param Condition The loop condition expression.
	 * @param Step The step statement expressions for the loop header.
	 * @param Body The loop body statements.
	 */
	ForNode(std::vector<std::unique_ptr<Node>> &Init, std::unique_ptr<Node> &Condition,
			std::vector<std::unique_ptr<Node>> &Step, std::vector<std::unique_ptr<Node>> &Body);

public:
	[[nodiscard]] NodeType Type() const override;

public:
	/**
	 * Getting the loop variable name
	 * @return The name of the loop variable
	 */
	[[nodiscard]] const std::string						   &VarName() const;

	/**
	 * Getting the loop start value
	 * @return The start value (inclusive)
	 */
	[[nodiscard]] int										Start() const;

	/**
	 * Getting the loop end value
	 * @return The end value (exclusive)
	 */
	[[nodiscard]] int										End() const;

	/**
	 * Getting the loop step value
	 * @return The step value
	 */
	[[nodiscard]] int										Step() const;

	/**
	 * Whether this node uses the dynamic header representation.
	 */
	[[nodiscard]] bool										HasDynamicHeader() const;

	/**
	 * Getting dynamic header initializer nodes.
	 */
	[[nodiscard]] const std::vector<std::unique_ptr<Node>> &Init() const;

	/**
	 * Getting dynamic header condition node.
	 */
	[[nodiscard]] const std::unique_ptr<Node>			   &Condition() const;

	/**
	 * Getting dynamic header step nodes.
	 */
	[[nodiscard]] const std::vector<std::unique_ptr<Node>> &StepNodes() const;

	/**
	 * Getting the loop body
	 * @return The node list of the loop body
	 */
	[[nodiscard]] const std::vector<std::unique_ptr<Node>> &Body() const;

	[[nodiscard]] std::unique_ptr<Node>						Clone() const override;

private:
	std::string						   _varName;
	int								   _start;
	int								   _end;
	int								   _step;
	bool							   _hasDynamicHeader = false;
	std::vector<std::unique_ptr<Node>> _init;
	std::unique_ptr<Node>			   _condition;
	std::vector<std::unique_ptr<Node>> _stepNodes;
	std::vector<std::unique_ptr<Node>> _body;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_FOR_H
