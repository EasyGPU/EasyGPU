/**
 * @file TextureStore.cpp
 * @brief Implementation of the texture store IR node.
 */

#include <IR/Node/TextureStore.h>

namespace GPU::IR::Node {
TextureStoreNode::TextureStoreNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
								   std::unique_ptr<Node> Value)
	: _textureName(std::move(TextureName)), _x(std::move(X)), _y(std::move(Y)), _value(std::move(Value)) {
}

TextureStoreNode::TextureStoreNode(std::string TextureName, std::unique_ptr<Node> X, std::unique_ptr<Node> Y,
								   std::unique_ptr<Node> Z, std::unique_ptr<Node> Value)
	: _textureName(std::move(TextureName)), _x(std::move(X)), _y(std::move(Y)), _z(std::move(Z)),
	  _value(std::move(Value)) {
}

NodeType TextureStoreNode::Type() const {
	return NodeType::TextureStore;
}

const std::string &TextureStoreNode::TextureName() const {
	return _textureName;
}

const Node *TextureStoreNode::X() const {
	return _x.get();
}

const Node *TextureStoreNode::Y() const {
	return _y.get();
}

const Node *TextureStoreNode::Z() const {
	return _z.get();
}

const Node *TextureStoreNode::Value() const {
	return _value.get();
}

std::unique_ptr<Node> TextureStoreNode::Clone() const {
	auto x = _x ? _x->Clone() : nullptr;
	auto y = _y ? _y->Clone() : nullptr;
	auto z = _z ? _z->Clone() : nullptr;
	auto value = _value ? _value->Clone() : nullptr;
	return std::make_unique<TextureStoreNode>(_textureName, std::move(x), std::move(y), std::move(z),
											  std::move(value));
}
} // namespace GPU::IR::Node
