#pragma once

/**
 * @file TextureLoad.h
 * @brief The node for imageLoad from a texture/image resource.
 */

#ifndef EASYGPU_TEXTURELOAD_H
#define EASYGPU_TEXTURELOAD_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>

namespace GPU::IR::Node {
/**
 * The node for loading one texel from a 2D or 3D image resource.
 */
class TextureLoadNode : public Node {
public:
	TextureLoadNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y);
	TextureLoadNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
					std::unique_ptr<Node> Z);

	[[nodiscard]] NodeType Type() const override;

	[[nodiscard]] const std::string &TextureName() const;
	[[nodiscard]] const Node		  *X() const;
	[[nodiscard]] const Node		  *Y() const;
	[[nodiscard]] const Node		  *Z() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string			  _textureName;
	std::unique_ptr<Node> _x;
	std::unique_ptr<Node> _y;
	std::unique_ptr<Node> _z;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_TEXTURELOAD_H
