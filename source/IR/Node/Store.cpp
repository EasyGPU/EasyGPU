/**
 * Store.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Node/Store.h>

namespace GPU::IR::Node {
    StoreNode::StoreNode(std::unique_ptr<Node> LHS, std::unique_ptr<Node> RHS) 
        : _lhs(std::move(LHS)), _rhs(std::move(RHS)) {
    }

    NodeType StoreNode::Type() const {
        return NodeType::Store;
    }

    const Node *StoreNode::LHS() const {
        return _lhs.get();
    }

    const Node *StoreNode::RHS() const {
        return _rhs.get();
    }

    std::unique_ptr<Node> StoreNode::Clone() const {
        std::unique_ptr<Node> lhsClone = _lhs ? _lhs->Clone() : nullptr;
        std::unique_ptr<Node> rhsClone = _rhs ? _rhs->Clone() : nullptr;
        return std::make_unique<StoreNode>(std::move(lhsClone), std::move(rhsClone));
    }
}
