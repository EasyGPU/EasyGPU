/**
 * SharedMemory.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */

#include <IR/Node/SharedMemory.h>

namespace GPU::IR::Node {
SharedMemoryNode::SharedMemoryNode(std::string Name, std::string Type, int Size)
	: _name(std::move(Name)), _type(std::move(Type)), _size(Size) {
}

NodeType SharedMemoryNode::Type() const {
	return NodeType::SharedMemory;
}

std::string SharedMemoryNode::VarName() const {
	return _name;
}

std::string SharedMemoryNode::VarType() const {
	return _type;
}

int SharedMemoryNode::Size() const {
	return _size;
}

std::unique_ptr<Node> SharedMemoryNode::Clone() const {
	return std::make_unique<SharedMemoryNode>(_name, _type, _size);
}
} // namespace GPU::IR::Node
