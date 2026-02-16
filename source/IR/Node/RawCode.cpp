/**
 * RawCode.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/RawCode.h>

namespace GPU::IR::Node {
    RawCodeNode::RawCodeNode(std::string Code)
        : _code(std::move(Code)) {
    }

    NodeType RawCodeNode::Type() const {
        return NodeType::RawCode;
    }

    const std::string &RawCodeNode::Code() const {
        return _code;
    }

    std::unique_ptr<Node> RawCodeNode::Clone() const {
        return std::make_unique<RawCodeNode>(_code);
    }
}
