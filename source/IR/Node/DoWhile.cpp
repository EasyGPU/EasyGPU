/**
 * DoWhile.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/DoWhile.h>

namespace GPU::IR::Node {
    DoWhileNode::DoWhileNode(std::vector<std::unique_ptr<Node>> &Body, std::unique_ptr<Node> &Condition)
        : _body(std::move(Body)),
          _condition(std::move(Condition)) {
    }

    NodeType DoWhileNode::Type() const {
        return NodeType::DoWhile;
    }

    const std::vector<std::unique_ptr<Node>> &DoWhileNode::Body() const {
        return _body;
    }

    const std::unique_ptr<Node> &DoWhileNode::Condition() const {
        return _condition;
    }

    std::unique_ptr<Node> DoWhileNode::Clone() const {
        std::vector<std::unique_ptr<Node>> bodyClone;
        bodyClone.reserve(_body.size());
        for (const auto &node : _body) {
            if (node) {
                bodyClone.push_back(node->Clone());
            } else {
                bodyClone.push_back(nullptr);
            }
        }

        std::unique_ptr<Node> conditionClone = _condition ? _condition->Clone() : nullptr;

        return std::make_unique<DoWhileNode>(bodyClone, conditionClone);
    }
}
