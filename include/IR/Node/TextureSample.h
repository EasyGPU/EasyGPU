#pragma once

/**
 * @file TextureSample.h
 * @brief The node for sampling a 2D sampled texture resource.
 */

#ifndef EASYGPU_TEXTURESAMPLE_H
#define EASYGPU_TEXTURESAMPLE_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>

namespace GPU::IR::Node {
/**
 * The node for texture() or textureLod() sampling.
 */
class TextureSampleNode : public Node {
public:
	TextureSampleNode(std::string TextureName, std::unique_ptr<Node> Coordinate);
	TextureSampleNode(std::string TextureName, std::unique_ptr<Node> Coordinate, std::unique_ptr<Node> Level);

	[[nodiscard]] NodeType Type() const override;

	[[nodiscard]] const std::string &TextureName() const;
	[[nodiscard]] const Node		  *Coordinate() const;
	[[nodiscard]] bool			   HasExplicitLevel() const;
	[[nodiscard]] const Node		  *Level() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string			  _textureName;
	std::unique_ptr<Node> _coordinate;
	std::unique_ptr<Node> _level;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_TEXTURESAMPLE_H
