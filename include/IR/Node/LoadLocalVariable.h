#pragma once

/**
 * @file LoadLocalVariable.h
 * @brief The load node for local variable.
 */

#ifndef EASYGPU_LOADLOCALVARIABLE_H
#define EASYGPU_LOADLOCALVARIABLE_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
/**
 * The load node for local variable
 */
class LoadLocalVariableNode : public LoadNode {
public:
	LoadLocalVariableNode(std::string Name);

public:
	[[nodiscard]] std::string			Unwrap() const override;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string _name;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_LOADLOCALVARIABLE_H