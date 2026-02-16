/**
 * Return.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/Return.h>

namespace GPU::IR::Node {
    ReturnNode::ReturnNode(std::unique_ptr<Node> Value) : _value(std::move(Value)) {
    }

    NodeType ReturnNode::Type() const {
        return NodeType::Return;
    }

    const Node *ReturnNode::Value() const {
        return _value.get();
    }

    std::unique_ptr<Node> ReturnNode::Clone() const {
        return std::make_unique<ReturnNode>(_value->Clone());
    }
}
