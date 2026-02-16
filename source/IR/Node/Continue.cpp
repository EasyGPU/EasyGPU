/**
 * ContinueNode.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */

#include <IR/Node/Continue.h>

namespace GPU::IR::Node {
    NodeType ContinueNode::Type() const {
        return NodeType::Continue;
    }

    std::unique_ptr<Node> ContinueNode::Clone() const {
        return std::make_unique<ContinueNode>();
    }
}
