/**
 * LocalVariableArray.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Node/LocalVariableArray.h>

#include <utility>

namespace GPU::IR::Node {
    LocalVariableArrayNode::LocalVariableArrayNode(std::string Name, std::string Type, int Size) : _name(std::move(Name)),
        _type(std::move(Type)), _size(Size) {
    }

    NodeType LocalVariableArrayNode::Type() const {
        return NodeType::LocalArray;
    }

    std::string LocalVariableArrayNode::VarName() const {
        return _name;
    }

    std::string LocalVariableArrayNode::VarType() const {
        return _type;
    }

    int LocalVariableArrayNode::Size() const {
        return _size;
    }

    std::unique_ptr<Node> LocalVariableArrayNode::Clone() const {
        return std::make_unique<LocalVariableArrayNode>(_name, _type, _size);
    }
}
