/**
 * Operation.h:
 *      @Descripiton    :   The operation node for the binary operation
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_OPERATION_H
#define EASYGPU_OPERATION_H

#include <IR/Node/Node.h>
#include <memory>

namespace GPU::IR::Node {
    /**
     * The operation code for the operation node
     */
    enum class OperationCode {
        // arithmetic
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        Neg,  // Unary minus

        // bitwise
        BitAnd,
        BitOr,
        BitXor,
        BitNot,
        Shl,
        Shr,

        // compare
        Less,
        Greater,
        Equal,
        NotEqual,
        LessEqual,
        GreaterEqual,

        // logical
        LogicalAnd,
        LogicalOr,
        LogicalNot,
    };

    /**
     * The operation node for the binary operation
     */
    class OperationNode : public Node {
    public:
        OperationNode(OperationCode Code, std::unique_ptr<Node> LHS, std::unique_ptr<Node> RHS = nullptr);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the operation code of this node
         * @return The operation code of the node
         */
        [[nodiscard]] OperationCode Code() const;

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
        OperationCode _code;
        std::unique_ptr<Node> _lhs;
        std::unique_ptr<Node> _rhs;
    };
}

#endif //EASYGPU_OPERATION_H
