/**
 * CompoundAssignment.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */

#include <IR/Node/CompoundAssignment.h>

namespace GPU::IR::Node {
    CompoundAssignmentNode::CompoundAssignmentNode(CompoundAssignmentCode code,
                                                   std::unique_ptr<Node>  lhs,
                                                   std::unique_ptr<Node>  rhs)
        : _code(code), _lhs(std::move(lhs)), _rhs(std::move(rhs)) {
    }

    NodeType CompoundAssignmentNode::Type() const {
        return NodeType::CompoundAssignment;
    }

    CompoundAssignmentCode CompoundAssignmentNode::Code() const {
        return _code;
    }

    const Node *CompoundAssignmentNode::LHS() const {
        return _lhs.get();
    }

    const Node *CompoundAssignmentNode::RHS() const {
        return _rhs.get();
    }

    std::unique_ptr<Node> CompoundAssignmentNode::Clone() const {
        return std::make_unique<CompoundAssignmentNode>(_code, _lhs->Clone(), _rhs->Clone());
    }
}
