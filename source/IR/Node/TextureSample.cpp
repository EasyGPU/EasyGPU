/**
 * @file TextureSample.cpp
 * @brief Implementation of the texture sample IR node.
 */

#include <IR/Node/TextureSample.h>

namespace GPU::IR::Node {
TextureSampleNode::TextureSampleNode(std::string TextureName, std::unique_ptr<Node> Coordinate)
	: _textureName(std::move(TextureName)), _coordinate(std::move(Coordinate)) {
}

TextureSampleNode::TextureSampleNode(std::string TextureName, std::unique_ptr<Node> Coordinate,
									 std::unique_ptr<Node> Level)
	: _textureName(std::move(TextureName)), _coordinate(std::move(Coordinate)), _level(std::move(Level)) {
}

NodeType TextureSampleNode::Type() const {
	return NodeType::TextureSample;
}

const std::string &TextureSampleNode::TextureName() const {
	return _textureName;
}

const Node *TextureSampleNode::Coordinate() const {
	return _coordinate.get();
}

bool TextureSampleNode::HasExplicitLevel() const {
	return _level != nullptr;
}

const Node *TextureSampleNode::Level() const {
	return _level.get();
}

std::unique_ptr<Node> TextureSampleNode::Clone() const {
	auto coordinate = _coordinate ? _coordinate->Clone() : nullptr;
	if (_level == nullptr) {
		return std::make_unique<TextureSampleNode>(_textureName, std::move(coordinate));
	}

	auto level = _level->Clone();
	return std::make_unique<TextureSampleNode>(_textureName, std::move(coordinate), std::move(level));
}
} // namespace GPU::IR::Node
