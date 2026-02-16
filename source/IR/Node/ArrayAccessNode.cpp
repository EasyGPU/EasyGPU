/**
 * ArrayAccessNode.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Node/ArrayAccess.h>

namespace GPU::IR::Node {
    ArrayAccessNode::ArrayAccessNode(std::unique_ptr<Node> Target,
                                     std::unique_ptr<Node> Index) : _target(std::move(Target)),
                                                                    _index(std::move(Index)) {
    }

    const Node *ArrayAccessNode::Index() const {
        return _index.get();
    }

    const Node *ArrayAccessNode::Target() const {
        return _target.get();
    }

    NodeType ArrayAccessNode::Type() const {
        return NodeType::ArrayAccess;
    }

    std::unique_ptr<Node> ArrayAccessNode::Clone() const {
        std::unique_ptr<Node> targetClone = _target ? _target->Clone() : nullptr;
        std::unique_ptr<Node> indexClone = _index ? _index->Clone() : nullptr;
        return std::make_unique<ArrayAccessNode>(std::move(targetClone), std::move(indexClone));
    }
}
