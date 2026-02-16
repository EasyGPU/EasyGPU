/**
 * Break.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/Break.h>

namespace GPU::IR::Node {
    NodeType BreakNode::Type() const {
        return NodeType::Break;
    }

    std::unique_ptr<Node> BreakNode::Clone() const {
        return std::make_unique<BreakNode>();
    }
}
