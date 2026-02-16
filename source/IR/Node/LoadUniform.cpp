/**
 * LoadUniform.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Node/LoadUniform.h>

namespace GPU::IR::Node {
    LoadUniformNode::LoadUniformNode(std::string Uniform) : _uniform(std::move(Uniform)) {
    }

    std::string LoadUniformNode::Unwarp() const {
        return _uniform;
    }
    
    std::unique_ptr<Node> LoadUniformNode::Clone() const {
        return std::make_unique<LoadUniformNode>(_uniform);
    }
}
