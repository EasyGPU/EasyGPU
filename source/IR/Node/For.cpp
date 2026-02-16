/**
 * For.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/For.h>

namespace GPU::IR::Node {
    ForNode::ForNode(const std::string &VarName, int Start, int End, int Step,
                     std::vector<std::unique_ptr<Node>> &Body)
        : _varName(VarName),
          _start(Start),
          _end(End),
          _step(Step),
          _body(std::move(Body)) {
    }

    NodeType ForNode::Type() const {
        return NodeType::For;
    }

    const std::string &ForNode::VarName() const {
        return _varName;
    }

    int ForNode::Start() const {
        return _start;
    }

    int ForNode::End() const {
        return _end;
    }

    int ForNode::Step() const {
        return _step;
    }

    const std::vector<std::unique_ptr<Node>> &ForNode::Body() const {
        return _body;
    }

    std::unique_ptr<Node> ForNode::Clone() const {
        std::vector<std::unique_ptr<Node>> bodyClone;
        bodyClone.reserve(_body.size());
        for (const auto &node : _body) {
            if (node) {
                bodyClone.push_back(node->Clone());
            } else {
                bodyClone.push_back(nullptr);
            }
        }

        return std::make_unique<ForNode>(_varName, _start, _end, _step, bodyClone);
    }
}
