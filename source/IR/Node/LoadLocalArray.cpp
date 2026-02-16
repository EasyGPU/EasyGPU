/**
 * LoadLocalArray.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/LoadLocalArray.h>

#include <format>
#include <utility>

namespace GPU::IR::Node {
    LoadLocalArrayNode::LoadLocalArrayNode(std::string Name) : _name(std::move(Name)) {
    }

    std::string LoadLocalArrayNode::Unwarp() const {
        return _name;
    }
    
    std::unique_ptr<Node> LoadLocalArrayNode::Clone() const {
        return std::make_unique<LoadLocalArrayNode>(_name);
    }
}
