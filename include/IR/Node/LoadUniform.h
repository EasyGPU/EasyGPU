#pragma once

/**
 * @file LoadUniform.h
 * @brief The uniform load node.
 */

#ifndef EASYGPU_LOADUNIFORM_H
#define EASYGPU_LOADUNIFORM_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
/**
 * The uniform node is the node to load a constant which may be captured by API from C++ side
 */
class LoadUniformNode : public LoadNode {
public:
	LoadUniformNode(std::string Uniform);

public:
	[[nodiscard]] std::string			Unwrap() const override;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string _uniform;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_LOADUNIFORM_H