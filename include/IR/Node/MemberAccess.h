/**
 * MemberAccess.h:
 *      @Descripiton    :   The node for member access
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_MEMBERACCESS_H
#define EASYGPU_MEMBERACCESS_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * The node for structure member access
     */
    class MemberAccessNode : public Node {
    public:
        MemberAccessNode(std::unique_ptr<Node> &LHS, std::unique_ptr<Node> &RHS);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the left hand side node of the node
         * @return The left hand side node of the node
         */
        [[nodiscard]] const Node * LHS() const;

        /**
         * Getting the right hand side node of the node
         * @return The right hand side node of the node
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

#endif //EASYGPU_MEMBERACCESS_H