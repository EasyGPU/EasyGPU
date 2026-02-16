/**
 * LocalVariable.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/LocalVariable.h>

#include <utility>

namespace GPU::IR::Node {
    LocalVariableNode::LocalVariableNode(std::string Name, std::string Type) 
        : _name(std::move(Name)), _type(std::move(Type)), _isExternal(false) {
    }
    
    LocalVariableNode::LocalVariableNode(std::string Name, std::string Type, bool IsExternal) 
        : _name(std::move(Name)), _type(std::move(Type)), _isExternal(IsExternal) {
    }

    std::string LocalVariableNode::VarName() const {
        return _name;
    }

    std::string LocalVariableNode::VarType() const {
        return _type;
    }
    
    bool LocalVariableNode::IsExternal() const {
        return _isExternal;
    }

    NodeType LocalVariableNode::Type() const {
        return NodeType::LocalVariable;
    }

    std::unique_ptr<Node> LocalVariableNode::Clone() const {
        auto clone = std::make_unique<LocalVariableNode>(_name, _type, _isExternal);
        return clone;
    }
}
