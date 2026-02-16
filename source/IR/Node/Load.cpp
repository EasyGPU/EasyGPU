/**
 * Load.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
    NodeType LoadNode::Type() const {
        return NodeType::Load;
    }
    
    // Clone() is pure virtual - implemented by derived classes
}
