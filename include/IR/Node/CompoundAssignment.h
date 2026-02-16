#pragma once

/**
 * CompoundAssignment.h:
 *      @Descripiton    :   The node for compound assignment
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_COMPOUNDASSIGNMENT_H
#define EASYGPU_COMPOUNDASSIGNMENT_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * Compound assignment operation codes
     * Represents atomic read-modify-write operations
     */
    enum class CompoundAssignmentCode {
        // Arithmetic compound assignments
        AddAssign, // +=
        SubAssign, // -=
        MulAssign, // *=
        DivAssign, // /=
        ModAssign, // %=

        // Bitwise compound assignments (integer types only)
        BitAndAssign, // &=
        BitOrAssign,  // |=
        BitXorAssign, // ^=
        ShlAssign,    // <<=
        ShrAssign,    // >>=
    };

    /**
     * The node for compound assignment
     */
    class CompoundAssignmentNode : public Node {
    public:
        /**
         * Construct compound assignment node
         * @param code   The compound assignment operation code
         * @param lhs    Addressable storage location (must be LoadLocalVariableNode or similar lvalue node)
         * @param rhs    Modifying expression (any value node)
         */
        CompoundAssignmentNode(CompoundAssignmentCode code,
                               std::unique_ptr<Node>  lhs,
                               std::unique_ptr<Node>  rhs);

        [[nodiscard]] NodeType Type() const override;

        /**
         * Getting the operation code of the node
         * @return The operation code of the node
         */
        [[nodiscard]] CompoundAssignmentCode Code() const;

        /**
         * Getting the left hand side node
         * @return The left hand side node
         */
        [[nodiscard]] const Node *LHS() const;

        /**
         * Getting the right hand side node
         * @return The right hand side node
         */
        [[nodiscard]] const Node *RHS() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        CompoundAssignmentCode _code;
        std::unique_ptr<Node>  _lhs;
        std::unique_ptr<Node>  _rhs;
    };
}

#endif //EASYGPU_COMPOUNDASSIGNMENT_H
