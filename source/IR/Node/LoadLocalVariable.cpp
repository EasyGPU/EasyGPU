/**
 * LoadLocalVariable.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/LoadLocalVariable.h>

#include <utility>


namespace GPU::IR::Node {
    LoadLocalVariableNode::LoadLocalVariableNode(std::string Name) : _name(std::move(Name)) {
    }

    std::string LoadLocalVariableNode::Unwarp() const {
        return _name;
    }
    
    std::unique_ptr<Node> LoadLocalVariableNode::Clone() const {
        return std::make_unique<LoadLocalVariableNode>(_name);
    }
}
