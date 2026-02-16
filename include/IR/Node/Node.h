/**
 * Node.h:
 *      @Descripiton    :   The base class for all the node type
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_NODE_H
#define EASYGPU_NODE_H

#include <memory>

namespace GPU::IR::Node {
    /**
     * The type of nodes
     */
    enum class NodeType {
        LocalVariable, LocalArray, Load, CallInst, Operation, Store, ArrayAccess, CompoundAssignment,
        Increment, MemberAccess, If, While, DoWhile, For, RawCode, Break, Continue, Return, Call
    };

    /**
     * The base class for all the nodes in the IR
     */
    class Node {
    public:
        Node() = default;

        virtual ~Node() = default;

    public:
        /**
         * Getting the type of the node
         * @return The type of this node
         */
        [[nodiscard]] virtual NodeType Type() const = 0;
        
        /**
         * Clone the node and its children
         * @return A deep copy of this node
         */
        [[nodiscard]] virtual std::unique_ptr<Node> Clone() const = 0;
    };
}

#endif //EASYGPU_NODE_H