#pragma once

/**
 * @file TextureStore.h
 * @brief The node for imageStore into a texture/image resource.
 */

#ifndef EASYGPU_TEXTURESTORE_H
#define EASYGPU_TEXTURESTORE_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>

namespace GPU::IR::Node {
/**
 * The node for storing one texel into a 2D or 3D image resource.
 */
class TextureStoreNode : public Node {
public:
	TextureStoreNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
					 std::unique_ptr<Node> Value);
	TextureStoreNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
					 std::unique_ptr<Node> Z, std::unique_ptr<Node> Value);

	[[nodiscard]] NodeType Type() const override;

	[[nodiscard]] const std::string &TextureName() const;
	[[nodiscard]] const Node		  *X() const;
	[[nodiscard]] const Node		  *Y() const;
	[[nodiscard]] const Node		  *Z() const;
	[[nodiscard]] const Node		  *Value() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string			  _textureName;
	std::unique_ptr<Node> _x;
	std::unique_ptr<Node> _y;
	std::unique_ptr<Node> _z;
	std::unique_ptr<Node> _value;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_TEXTURESTORE_H
