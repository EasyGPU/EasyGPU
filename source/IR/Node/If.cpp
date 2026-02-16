/**
 * If.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/If.h>

namespace GPU::IR::Node {
    IfNode::IfNode(std::vector<std::unique_ptr<Node>> &Do, std::unique_ptr<Node> &Condition,
                   std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node>>>> &Elifs,
                   std::vector<std::unique_ptr<Node>> &Else)
        : _do(std::move(Do)),
          _condition(std::move(Condition)),
          _elifs(std::move(Elifs)),
          _else(std::move(Else)) {
    }

    NodeType IfNode::Type() const {
        return NodeType::If;
    }

    const std::vector<std::unique_ptr<Node>> &IfNode::Do() const {
        return _do;
    }

    const std::unique_ptr<Node> &IfNode::Condition() const {
        return _condition;
    }

    const std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node>>>> &IfNode::Elifs() const {
        return _elifs;
    }

    const std::vector<std::unique_ptr<Node>> &IfNode::Else() const {
        return _else;
    }

    std::unique_ptr<Node> IfNode::Clone() const {
        std::vector<std::unique_ptr<Node>> doClone;
        doClone.reserve(_do.size());
        for (const auto &node : _do) {
            if (node) {
                doClone.push_back(node->Clone());
            } else {
                doClone.push_back(nullptr);
            }
        }

        std::unique_ptr<Node> conditionClone = _condition ? _condition->Clone() : nullptr;

        std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node>>>> elifsClone;
        elifsClone.reserve(_elifs.size());
        for (const auto &elif : _elifs) {
            std::unique_ptr<Node> elifConditionClone = elif.first ? elif.first->Clone() : nullptr;
            std::vector<std::unique_ptr<Node>> elifBodyClone;
            elifBodyClone.reserve(elif.second.size());
            for (const auto &node : elif.second) {
                if (node) {
                    elifBodyClone.push_back(node->Clone());
                } else {
                    elifBodyClone.push_back(nullptr);
                }
            }
            elifsClone.push_back(std::make_pair(std::move(elifConditionClone), std::move(elifBodyClone)));
        }

        std::vector<std::unique_ptr<Node>> elseClone;
        elseClone.reserve(_else.size());
        for (const auto &node : _else) {
            if (node) {
                elseClone.push_back(node->Clone());
            } else {
                elseClone.push_back(nullptr);
            }
        }

        return std::make_unique<IfNode>(doClone, conditionClone, elifsClone, elseClone);
    }
}
