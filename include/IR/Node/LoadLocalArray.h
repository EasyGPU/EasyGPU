#pragma once

/**
 * @file LoadLocalArray.h
 * @brief The load node for local array.
 */

#ifndef EASYGPU_LOADLOCALARRAY_H
#define EASYGPU_LOADLOCALARRAY_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
/**
 * The load node for local array
 */
class LoadLocalArrayNode : public LoadNode {
public:
	LoadLocalArrayNode(std::string Name);

public:
	[[nodiscard]] std::string			Unwrap() const override;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string _name;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_LOADLOCALARRAY_H