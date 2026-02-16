/**
 * MemberAccess.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */

#include <IR/Node/MemberAccess.h>

namespace GPU::IR::Node {
    MemberAccessNode::MemberAccessNode(std::unique_ptr<Node> &LHS, std::unique_ptr<Node> &RHS) : _lhs(std::move(LHS)),
        _rhs(std::move(RHS)) {
    }

    NodeType MemberAccessNode::Type() const {
        return NodeType::MemberAccess;
    }

    const Node *MemberAccessNode::LHS() const {
        return _lhs.get();
    }

    const Node *MemberAccessNode::RHS() const {
        return _rhs.get();
    }
    
    std::unique_ptr<Node> MemberAccessNode::Clone() const {
        std::unique_ptr<Node> lhsClone = _lhs ? _lhs->Clone() : nullptr;
        std::unique_ptr<Node> rhsClone = _rhs ? _rhs->Clone() : nullptr;
        return std::make_unique<MemberAccessNode>(lhsClone, rhsClone);
    }
}
