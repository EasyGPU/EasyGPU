/**
 * While.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/While.h>

namespace GPU::IR::Node {
    WhileNode::WhileNode(std::unique_ptr<Node> &Condition, std::vector<std::unique_ptr<Node> > &Body)
        : _condition(std::move(Condition)),
          _body(std::move(Body)) {
    }

    NodeType WhileNode::Type() const {
        return NodeType::While;
    }

    const std::unique_ptr<Node> &WhileNode::Condition() const {
        return _condition;
    }

    const std::vector<std::unique_ptr<Node> > &WhileNode::Body() const {
        return _body;
    }

    std::unique_ptr<Node> WhileNode::Clone() const {
        std::unique_ptr<Node> conditionClone = _condition ? _condition->Clone() : nullptr;

        std::vector<std::unique_ptr<Node>> bodyClone;
        bodyClone.reserve(_body.size());
        for (const auto &node : _body) {
            if (node) {
                bodyClone.push_back(node->Clone());
            } else {
                bodyClone.push_back(nullptr);
            }
        }

        return std::make_unique<WhileNode>(conditionClone, bodyClone);
    }
}
