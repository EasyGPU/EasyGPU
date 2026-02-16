/**
 * Increment.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */

#include <IR/Node/Increment.h>

namespace GPU::IR::Node {
    IncrementNode::IncrementNode(IncrementDirection Direction,
                      std::unique_ptr<Node> Target,
                      bool IsPrefix)
            : _direction(Direction), _target(std::move(Target)), _isPrefix(IsPrefix) {}

    NodeType IncrementNode::Type() const {
        return NodeType::Increment;
    }

    IncrementDirection IncrementNode::Direction() const {
        return _direction;
    }

    const Node* IncrementNode::Target() const {
        return _target.get();
    }

    bool IncrementNode::IsPrefix() const {
        return _isPrefix;
    }

    std::unique_ptr<Node> IncrementNode::Clone() const {
        return std::make_unique<IncrementNode>(_direction, _target->Clone(), _isPrefix);
    }
}