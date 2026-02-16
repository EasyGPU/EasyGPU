/**
 * LocalVariable.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/LocalVariable.h>

#include <utility>

namespace GPU::IR::Node {
    LocalVariableNode::LocalVariableNode(std::string Name, std::string Type) : _name(std::move(Name)), _type(std::move(Type)) {
    }

    std::string LocalVariableNode::VarName() const {
        return _name;
    }

    std::string LocalVariableNode::VarType() const {
        return _type;
    }

    NodeType LocalVariableNode::Type() const {
        return NodeType::LocalVariable;
    }

    std::unique_ptr<Node> LocalVariableNode::Clone() const {
        return std::make_unique<LocalVariableNode>(_name, _type);
    }
}
