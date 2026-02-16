/**
 * Call.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */

#include <IR/Node/Call.h>

namespace GPU::IR::Node {
    CallNode::CallNode(std::string FuncName, std::vector<std::unique_ptr<Node>> Arguments)
        : _funcName(std::move(FuncName)), _arguments(std::move(Arguments)) {
    }

    NodeType CallNode::Type() const {
        return NodeType::Call;
    }

    const std::string &CallNode::FuncName() const {
        return _funcName;
    }

    const std::vector<std::unique_ptr<Node>> &CallNode::Arguments() const {
        return _arguments;
    }

    std::unique_ptr<Node> CallNode::Clone() const {
        std::vector<std::unique_ptr<Node>> clonedArgs;
        clonedArgs.reserve(_arguments.size());
        for (const auto &arg : _arguments) {
            clonedArgs.push_back(arg->Clone());
        }
        return std::make_unique<CallNode>(_funcName, std::move(clonedArgs));
    }
}
