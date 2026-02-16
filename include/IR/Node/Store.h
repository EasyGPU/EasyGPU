/**
 * Store.h:
 *      @Descripiton    :   The store node
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_STORE_H
#define EASYGPU_STORE_H

#include <IR/Node/Node.h>
#include <memory>

namespace GPU::IR::Node {
    /**
     * The node for store instruction
     */
    class StoreNode : public Node {
    public:
        StoreNode(std::unique_ptr<Node> LHS, std::unique_ptr<Node> RHS);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the left hand side node of the operation node
         * @return The left hand side node of the operation node
         */
        [[nodiscard]] const Node * LHS() const;

        /**
         * Getting the right hand side node of the operation node
         * @return The right hand side node of the operation node, if it
         * is unary, it will return nullptr
         */
        [[nodiscard]] const Node * RHS() const;

        /**
         * Clone this node and its children
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::unique_ptr<Node> _lhs;
        std::unique_ptr<Node> _rhs;
    };
}

#endif //EASYGPU_STORE_H
