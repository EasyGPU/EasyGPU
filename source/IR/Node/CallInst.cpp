/**
 * CallInst.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Node/CallInst.h>

#include <utility>

namespace GPU::IR::Node {
    IntrinsicCallNode::IntrinsicCallNode(std::string Name, std::vector<std::unique_ptr<Node *>> Parameter) : _name(std::move(Name)),
        _parameter(std::move(Parameter)) {
    }

    NodeType IntrinsicCallNode::Type() const {
        return NodeType::CallInst;
    }

    std::string_view IntrinsicCallNode::Name() const {
        return _name;
    }

    const std::vector<std::unique_ptr<Node *> > &IntrinsicCallNode::Parameter() const {
        return _parameter;
    }

    std::unique_ptr<Node> IntrinsicCallNode::Clone() const {
        std::vector<std::unique_ptr<Node *>> clonedParams;
        clonedParams.reserve(_parameter.size());
        for (const auto &param : _parameter) {
            if (param && *param) {
                clonedParams.push_back(std::make_unique<Node *>((*param)->Clone().release()));
            } else {
                clonedParams.push_back(nullptr);
            }
        }
        return std::make_unique<IntrinsicCallNode>(_name, std::move(clonedParams));
    }
}
