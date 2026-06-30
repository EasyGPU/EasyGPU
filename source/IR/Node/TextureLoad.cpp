/**
 * @file TextureLoad.cpp
 * @brief Implementation of the texture load IR node.
 */

#include <IR/Node/TextureLoad.h>

namespace GPU::IR::Node {
TextureLoadNode::TextureLoadNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y)
	: _textureName(std::move(TextureName)), _x(std::move(X)), _y(std::move(Y)) {
}

TextureLoadNode::TextureLoadNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
								 std::unique_ptr<Node> Z)
	: _textureName(std::move(TextureName)), _x(std::move(X)), _y(std::move(Y)), _z(std::move(Z)) {
}

NodeType TextureLoadNode::Type() const {
	return NodeType::TextureLoad;
}

const std::string &TextureLoadNode::TextureName() const {
	return _textureName;
}

const Node *TextureLoadNode::X() const {
	return _x.get();
}

const Node *TextureLoadNode::Y() const {
	return _y.get();
}

const Node *TextureLoadNode::Z() const {
	return _z.get();
}

std::unique_ptr<Node> TextureLoadNode::Clone() const {
	auto x = _x ? _x->Clone() : nullptr;
	auto y = _y ? _y->Clone() : nullptr;
	auto z = _z ? _z->Clone() : nullptr;
	return std::make_unique<TextureLoadNode>(_textureName, std::move(x), std::move(y), std::move(z));
}
} // namespace GPU::IR::Node
