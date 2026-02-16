/**
 * Operation.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/Operation.h>

namespace GPU::IR::Node {
    OperationNode::OperationNode(const OperationCode Code, std::unique_ptr<Node> LHS, std::unique_ptr<Node> RHS) 
        : _code(Code), _lhs(std::move(LHS)), _rhs(std::move(RHS)) {
    }

    NodeType OperationNode::Type() const {
        return NodeType::Operation;
    }

    OperationCode OperationNode::Code() const {
        return _code;
    }

    const Node *OperationNode::LHS() const {
        return _lhs.get();
    }

    const Node *OperationNode::RHS() const {
        return _rhs.get();
    }
    
    std::unique_ptr<Node> OperationNode::Clone() const {
        std::unique_ptr<Node> lhsClone = _lhs ? _lhs->Clone() : nullptr;
        std::unique_ptr<Node> rhsClone = _rhs ? _rhs->Clone() : nullptr;
        return std::make_unique<OperationNode>(_code, std::move(lhsClone), std::move(rhsClone));
    }
}
