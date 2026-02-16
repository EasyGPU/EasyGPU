#pragma once

/**
 * ExprIVector.h:
 *      @Descripiton    :   The specified expression API for int vector with swizzle access
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_EXPRIVECTOR_H
#define EASYGPU_EXPRIVECTOR_H

#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <Utility/Vec.h>

#include <format>

namespace GPU::IR::Value {
    // Swizzle macros for int vector expressions
    // Single component access returns Expr<int>
#define EXPR_IVEC_MEM(n) Expr<int> n() && { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<int>(std::make_unique<Node::LoadUniformNode>(std::format("{}.{}", exprStr, #n))); \
}

    // 2-component swizzle
#define EXPR_IVEC_SWZ2(n) Expr<Math::IVec2> n() && { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::IVec2>(std::make_unique<Node::LoadUniformNode>(std::format("({}).{}", exprStr, #n))); \
}

    // 3-component swizzle
#define EXPR_IVEC_SWZ3(n) Expr<Math::IVec3> n() && { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::IVec3>(std::make_unique<Node::LoadUniformNode>(std::format("{}.{}", exprStr, #n))); \
}

    // 4-component swizzle
#define EXPR_IVEC_SWZ4(n) Expr<Math::IVec4> n() && { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::IVec4>(std::make_unique<Node::LoadUniformNode>(std::format("{}.{}", exprStr, #n))); \
}

    // Specialization for IVec2 expressions with swizzle access
    template<>
    class Expr<Math::IVec2> : public ExprBase {
    public:
        using ValueType = Math::IVec2;
        using ElementType_t = int;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::IVec2>& var);
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) & = delete;

        Expr<int> operator[](ExprBase index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<int> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_IVEC_MEM(x) EXPR_IVEC_MEM(y)

        // 2-component swizzles
        EXPR_IVEC_SWZ2(xx) EXPR_IVEC_SWZ2(xy) EXPR_IVEC_SWZ2(yx) EXPR_IVEC_SWZ2(yy)
        /* clang-format on */

    public:
        // Arithmetic operations
        friend Expr<Math::IVec2> operator+(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator-(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator*(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator/(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Scalar operations with int literal
        friend Expr<Math::IVec2> operator*(Expr<Math::IVec2> lhs, int rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec2> operator*(int lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec2> operator/(Expr<Math::IVec2> lhs, int rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec2> operator+(Expr<Math::IVec2> lhs, int rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec2> operator+(int lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec2> operator-(Expr<Math::IVec2> lhs, int rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec2> operator-(int lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Scalar operations with Var<int>
        friend Expr<Math::IVec2> operator*(Expr<Math::IVec2> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec2> operator*(const Var<int>& lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec2> operator/(Expr<Math::IVec2> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec2> operator+(Expr<Math::IVec2> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec2> operator+(const Var<int>& lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec2> operator-(Expr<Math::IVec2> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec2> operator-(const Var<int>& lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Load(), rhs.Release()));
        }

        // Bitwise operations
        friend Expr<Math::IVec2> operator&(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator|(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator^(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator~(Expr<Math::IVec2> val) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitNot, val.Release()));
        }

        friend Expr<Math::IVec2> operator<<(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shl, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec2> operator>>(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<Math::IVec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shr, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::IVec2> lhs, Expr<Math::IVec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for IVec3 expressions with swizzle access
    template<>
    class Expr<Math::IVec3> : public ExprBase {
    public:
        using ValueType = Math::IVec3;
        using ElementType_t = int;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::IVec3>& var);
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) & = delete;

        Expr<int> operator[](ExprBase index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<int> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_IVEC_MEM(x) EXPR_IVEC_MEM(y) EXPR_IVEC_MEM(z)

        // 2-component swizzles
        EXPR_IVEC_SWZ2(xx) EXPR_IVEC_SWZ2(xy) EXPR_IVEC_SWZ2(xz)
        EXPR_IVEC_SWZ2(yx) EXPR_IVEC_SWZ2(yy) EXPR_IVEC_SWZ2(yz)
        EXPR_IVEC_SWZ2(zx) EXPR_IVEC_SWZ2(zy) EXPR_IVEC_SWZ2(zz)

        // 3-component swizzles
        EXPR_IVEC_SWZ3(xxx) EXPR_IVEC_SWZ3(xxy) EXPR_IVEC_SWZ3(xxz)
        EXPR_IVEC_SWZ3(xyx) EXPR_IVEC_SWZ3(xyy) EXPR_IVEC_SWZ3(xyz)
        EXPR_IVEC_SWZ3(xzx) EXPR_IVEC_SWZ3(xzy) EXPR_IVEC_SWZ3(xzz)

        EXPR_IVEC_SWZ3(yxx) EXPR_IVEC_SWZ3(yxy) EXPR_IVEC_SWZ3(yxz)
        EXPR_IVEC_SWZ3(yyx) EXPR_IVEC_SWZ3(yyy) EXPR_IVEC_SWZ3(yyz)
        EXPR_IVEC_SWZ3(yzx) EXPR_IVEC_SWZ3(yzy) EXPR_IVEC_SWZ3(yzz)

        EXPR_IVEC_SWZ3(zxx) EXPR_IVEC_SWZ3(zxy) EXPR_IVEC_SWZ3(zxz)
        EXPR_IVEC_SWZ3(zyx) EXPR_IVEC_SWZ3(zyy) EXPR_IVEC_SWZ3(zyz)
        EXPR_IVEC_SWZ3(zzx) EXPR_IVEC_SWZ3(zzy) EXPR_IVEC_SWZ3(zzz)
        /* clang-format on */

    public:
        // Arithmetic operations
        friend Expr<Math::IVec3> operator+(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator-(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator*(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator/(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Scalar operations with int literal
        friend Expr<Math::IVec3> operator*(Expr<Math::IVec3> lhs, int rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec3> operator*(int lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec3> operator/(Expr<Math::IVec3> lhs, int rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec3> operator+(Expr<Math::IVec3> lhs, int rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec3> operator+(int lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec3> operator-(Expr<Math::IVec3> lhs, int rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec3> operator-(int lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Scalar operations with Var<int>
        friend Expr<Math::IVec3> operator*(Expr<Math::IVec3> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec3> operator*(const Var<int>& lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec3> operator/(Expr<Math::IVec3> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec3> operator+(Expr<Math::IVec3> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec3> operator+(const Var<int>& lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec3> operator-(Expr<Math::IVec3> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec3> operator-(const Var<int>& lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Load(), rhs.Release()));
        }

        // Bitwise operations
        friend Expr<Math::IVec3> operator&(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator|(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator^(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator~(Expr<Math::IVec3> val) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitNot, val.Release()));
        }

        friend Expr<Math::IVec3> operator<<(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shl, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec3> operator>>(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<Math::IVec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shr, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::IVec3> lhs, Expr<Math::IVec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for IVec4 expressions with swizzle access
    template<>
    class Expr<Math::IVec4> : public ExprBase {
    public:
        using ValueType = Math::IVec4;
        using ElementType_t = int;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::IVec4>& var);
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<int> operator[](IndexType index) & = delete;

        Expr<int> operator[](ExprBase index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<int> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) && {
            return Expr<int>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<int> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_IVEC_MEM(x) EXPR_IVEC_MEM(y) EXPR_IVEC_MEM(z) EXPR_IVEC_MEM(w)

        // 2-component swizzles (16)
        EXPR_IVEC_SWZ2(xx) EXPR_IVEC_SWZ2(xy) EXPR_IVEC_SWZ2(xz) EXPR_IVEC_SWZ2(xw)
        EXPR_IVEC_SWZ2(yx) EXPR_IVEC_SWZ2(yy) EXPR_IVEC_SWZ2(yz) EXPR_IVEC_SWZ2(yw)
        EXPR_IVEC_SWZ2(zx) EXPR_IVEC_SWZ2(zy) EXPR_IVEC_SWZ2(zz) EXPR_IVEC_SWZ2(zw)
        EXPR_IVEC_SWZ2(wx) EXPR_IVEC_SWZ2(wy) EXPR_IVEC_SWZ2(wz) EXPR_IVEC_SWZ2(ww)

        // 3-component swizzles (64) - Partial
        EXPR_IVEC_SWZ3(xxx) EXPR_IVEC_SWZ3(xxy) EXPR_IVEC_SWZ3(xxz) EXPR_IVEC_SWZ3(xxw) EXPR_IVEC_SWZ3(xyx) EXPR_IVEC_SWZ3(xyy) EXPR_IVEC_SWZ3(xyz) EXPR_IVEC_SWZ3(xyw)
        EXPR_IVEC_SWZ3(xzx) EXPR_IVEC_SWZ3(xzy) EXPR_IVEC_SWZ3(xzz) EXPR_IVEC_SWZ3(xzw) EXPR_IVEC_SWZ3(xwx) EXPR_IVEC_SWZ3(xwy) EXPR_IVEC_SWZ3(xwz) EXPR_IVEC_SWZ3(xww)

        EXPR_IVEC_SWZ3(yxx) EXPR_IVEC_SWZ3(yxy) EXPR_IVEC_SWZ3(yxz) EXPR_IVEC_SWZ3(yxw) EXPR_IVEC_SWZ3(yyx) EXPR_IVEC_SWZ3(yyy) EXPR_IVEC_SWZ3(yyz) EXPR_IVEC_SWZ3(yyw)
        EXPR_IVEC_SWZ3(yzx) EXPR_IVEC_SWZ3(yzy) EXPR_IVEC_SWZ3(yzz) EXPR_IVEC_SWZ3(yzw) EXPR_IVEC_SWZ3(ywx) EXPR_IVEC_SWZ3(ywy) EXPR_IVEC_SWZ3(ywz) EXPR_IVEC_SWZ3(yww)

        EXPR_IVEC_SWZ3(zxx) EXPR_IVEC_SWZ3(zxy) EXPR_IVEC_SWZ3(zxz) EXPR_IVEC_SWZ3(zxw) EXPR_IVEC_SWZ3(zyx) EXPR_IVEC_SWZ3(zyy) EXPR_IVEC_SWZ3(zyz) EXPR_IVEC_SWZ3(zyw)
        EXPR_IVEC_SWZ3(zzx) EXPR_IVEC_SWZ3(zzy) EXPR_IVEC_SWZ3(zzz) EXPR_IVEC_SWZ3(zzw) EXPR_IVEC_SWZ3(zwx) EXPR_IVEC_SWZ3(zwy) EXPR_IVEC_SWZ3(zwz) EXPR_IVEC_SWZ3(zww)

        EXPR_IVEC_SWZ3(wxx) EXPR_IVEC_SWZ3(wxy) EXPR_IVEC_SWZ3(wxz) EXPR_IVEC_SWZ3(wxw) EXPR_IVEC_SWZ3(wyx) EXPR_IVEC_SWZ3(wyy) EXPR_IVEC_SWZ3(wyz) EXPR_IVEC_SWZ3(wyw)
        EXPR_IVEC_SWZ3(wzx) EXPR_IVEC_SWZ3(wzy) EXPR_IVEC_SWZ3(wzz) EXPR_IVEC_SWZ3(wzw) EXPR_IVEC_SWZ3(wwx) EXPR_IVEC_SWZ3(wwy) EXPR_IVEC_SWZ3(wwz) EXPR_IVEC_SWZ3(www)

        // 4-component swizzles (256) - Partial
        EXPR_IVEC_SWZ4(xxxx) EXPR_IVEC_SWZ4(xxxy) EXPR_IVEC_SWZ4(xxxz) EXPR_IVEC_SWZ4(xxxw) EXPR_IVEC_SWZ4(xxyx) EXPR_IVEC_SWZ4(xxyy) EXPR_IVEC_SWZ4(xxyz) EXPR_IVEC_SWZ4(xxyw)
        EXPR_IVEC_SWZ4(xxzx) EXPR_IVEC_SWZ4(xxzy) EXPR_IVEC_SWZ4(xxzz) EXPR_IVEC_SWZ4(xxzw) EXPR_IVEC_SWZ4(xxwx) EXPR_IVEC_SWZ4(xxwy) EXPR_IVEC_SWZ4(xxwz) EXPR_IVEC_SWZ4(xxww)

        EXPR_IVEC_SWZ4(xyxx) EXPR_IVEC_SWZ4(xyxy) EXPR_IVEC_SWZ4(xyxz) EXPR_IVEC_SWZ4(xyxw) EXPR_IVEC_SWZ4(xyyx) EXPR_IVEC_SWZ4(xyyy) EXPR_IVEC_SWZ4(xyyz) EXPR_IVEC_SWZ4(xyyw)
        EXPR_IVEC_SWZ4(xyzx) EXPR_IVEC_SWZ4(xyzy) EXPR_IVEC_SWZ4(xyzz) EXPR_IVEC_SWZ4(xyzw) EXPR_IVEC_SWZ4(xywx) EXPR_IVEC_SWZ4(xywy) EXPR_IVEC_SWZ4(xywz) EXPR_IVEC_SWZ4(xyww)

        EXPR_IVEC_SWZ4(xzxx) EXPR_IVEC_SWZ4(xzxy) EXPR_IVEC_SWZ4(xzxz) EXPR_IVEC_SWZ4(xzxw) EXPR_IVEC_SWZ4(xzyx) EXPR_IVEC_SWZ4(xzyy) EXPR_IVEC_SWZ4(xzyz) EXPR_IVEC_SWZ4(xzyw)
        EXPR_IVEC_SWZ4(xzzx) EXPR_IVEC_SWZ4(xzzy) EXPR_IVEC_SWZ4(xzzz) EXPR_IVEC_SWZ4(xzzw) EXPR_IVEC_SWZ4(xzwx) EXPR_IVEC_SWZ4(xzwy) EXPR_IVEC_SWZ4(xzwz) EXPR_IVEC_SWZ4(xzww)

        EXPR_IVEC_SWZ4(xwxx) EXPR_IVEC_SWZ4(xwxy) EXPR_IVEC_SWZ4(xwxz) EXPR_IVEC_SWZ4(xwxw) EXPR_IVEC_SWZ4(xwyx) EXPR_IVEC_SWZ4(xwyy) EXPR_IVEC_SWZ4(xwyz) EXPR_IVEC_SWZ4(xwyw)
        EXPR_IVEC_SWZ4(xwzx) EXPR_IVEC_SWZ4(xwzy) EXPR_IVEC_SWZ4(xwzz) EXPR_IVEC_SWZ4(xwzw) EXPR_IVEC_SWZ4(xwwx) EXPR_IVEC_SWZ4(xwwy) EXPR_IVEC_SWZ4(xwwz) EXPR_IVEC_SWZ4(xwww)

        EXPR_IVEC_SWZ4(yxxx) EXPR_IVEC_SWZ4(yxxy) EXPR_IVEC_SWZ4(yxxz) EXPR_IVEC_SWZ4(yxxw) EXPR_IVEC_SWZ4(yxyx) EXPR_IVEC_SWZ4(yxyy) EXPR_IVEC_SWZ4(yxyz) EXPR_IVEC_SWZ4(yxyw)
        EXPR_IVEC_SWZ4(yxzx) EXPR_IVEC_SWZ4(yxzy) EXPR_IVEC_SWZ4(yxzz) EXPR_IVEC_SWZ4(yxzw) EXPR_IVEC_SWZ4(yxwx) EXPR_IVEC_SWZ4(yxwy) EXPR_IVEC_SWZ4(yxwz) EXPR_IVEC_SWZ4(yxww)

        EXPR_IVEC_SWZ4(yyxx) EXPR_IVEC_SWZ4(yyxy) EXPR_IVEC_SWZ4(yyxz) EXPR_IVEC_SWZ4(yyxw) EXPR_IVEC_SWZ4(yyyx) EXPR_IVEC_SWZ4(yyyy) EXPR_IVEC_SWZ4(yyyz) EXPR_IVEC_SWZ4(yyyw)
        EXPR_IVEC_SWZ4(yyzx) EXPR_IVEC_SWZ4(yyzy) EXPR_IVEC_SWZ4(yyzz) EXPR_IVEC_SWZ4(yyzw) EXPR_IVEC_SWZ4(yywx) EXPR_IVEC_SWZ4(yywy) EXPR_IVEC_SWZ4(yywz) EXPR_IVEC_SWZ4(yyww)

        EXPR_IVEC_SWZ4(yzxx) EXPR_IVEC_SWZ4(yzxy) EXPR_IVEC_SWZ4(yzxz) EXPR_IVEC_SWZ4(yzxw) EXPR_IVEC_SWZ4(yzyx) EXPR_IVEC_SWZ4(yzyy) EXPR_IVEC_SWZ4(yzyz) EXPR_IVEC_SWZ4(yzyw)
        EXPR_IVEC_SWZ4(yzzx) EXPR_IVEC_SWZ4(yzzy) EXPR_IVEC_SWZ4(yzzz) EXPR_IVEC_SWZ4(yzzw) EXPR_IVEC_SWZ4(yzwx) EXPR_IVEC_SWZ4(yzwy) EXPR_IVEC_SWZ4(yzwz) EXPR_IVEC_SWZ4(yzww)

        EXPR_IVEC_SWZ4(ywxx) EXPR_IVEC_SWZ4(ywxy) EXPR_IVEC_SWZ4(ywxz) EXPR_IVEC_SWZ4(ywxw) EXPR_IVEC_SWZ4(ywyx) EXPR_IVEC_SWZ4(ywyy) EXPR_IVEC_SWZ4(ywyz) EXPR_IVEC_SWZ4(ywyw)
        EXPR_IVEC_SWZ4(ywzx) EXPR_IVEC_SWZ4(ywzy) EXPR_IVEC_SWZ4(ywzz) EXPR_IVEC_SWZ4(ywzw) EXPR_IVEC_SWZ4(ywwx) EXPR_IVEC_SWZ4(ywwy) EXPR_IVEC_SWZ4(ywwz) EXPR_IVEC_SWZ4(ywww)

        EXPR_IVEC_SWZ4(zxxx) EXPR_IVEC_SWZ4(zxxy) EXPR_IVEC_SWZ4(zxxz) EXPR_IVEC_SWZ4(zxxw) EXPR_IVEC_SWZ4(zxyx) EXPR_IVEC_SWZ4(zxyy) EXPR_IVEC_SWZ4(zxyz) EXPR_IVEC_SWZ4(zxyw)
        EXPR_IVEC_SWZ4(zxzx) EXPR_IVEC_SWZ4(zxzy) EXPR_IVEC_SWZ4(zxzz) EXPR_IVEC_SWZ4(zxzw) EXPR_IVEC_SWZ4(zxwx) EXPR_IVEC_SWZ4(zxwy) EXPR_IVEC_SWZ4(zxwz) EXPR_IVEC_SWZ4(zxww)

        EXPR_IVEC_SWZ4(zyxx) EXPR_IVEC_SWZ4(zyxy) EXPR_IVEC_SWZ4(zyxz) EXPR_IVEC_SWZ4(zyxw) EXPR_IVEC_SWZ4(zyyx) EXPR_IVEC_SWZ4(zyyy) EXPR_IVEC_SWZ4(zyyz) EXPR_IVEC_SWZ4(zyyw)
        EXPR_IVEC_SWZ4(zyzx) EXPR_IVEC_SWZ4(zyzy) EXPR_IVEC_SWZ4(zyzz) EXPR_IVEC_SWZ4(zyzw) EXPR_IVEC_SWZ4(zywx) EXPR_IVEC_SWZ4(zywy) EXPR_IVEC_SWZ4(zywz) EXPR_IVEC_SWZ4(zyww)

        EXPR_IVEC_SWZ4(zzxx) EXPR_IVEC_SWZ4(zzxy) EXPR_IVEC_SWZ4(zzxz) EXPR_IVEC_SWZ4(zzxw) EXPR_IVEC_SWZ4(zzyx) EXPR_IVEC_SWZ4(zzyy) EXPR_IVEC_SWZ4(zzyz) EXPR_IVEC_SWZ4(zzyw)
        EXPR_IVEC_SWZ4(zzzx) EXPR_IVEC_SWZ4(zzzy) EXPR_IVEC_SWZ4(zzzz) EXPR_IVEC_SWZ4(zzzw) EXPR_IVEC_SWZ4(zzwx) EXPR_IVEC_SWZ4(zzwy) EXPR_IVEC_SWZ4(zzwz) EXPR_IVEC_SWZ4(zzww)

        EXPR_IVEC_SWZ4(zwxx) EXPR_IVEC_SWZ4(zwxy) EXPR_IVEC_SWZ4(zwxz) EXPR_IVEC_SWZ4(zwxw) EXPR_IVEC_SWZ4(zwyx) EXPR_IVEC_SWZ4(zwyy) EXPR_IVEC_SWZ4(zwyz) EXPR_IVEC_SWZ4(zwyw)
        EXPR_IVEC_SWZ4(zwzx) EXPR_IVEC_SWZ4(zwzy) EXPR_IVEC_SWZ4(zwzz) EXPR_IVEC_SWZ4(zwzw) EXPR_IVEC_SWZ4(zwwx) EXPR_IVEC_SWZ4(zwwy) EXPR_IVEC_SWZ4(zwwz) EXPR_IVEC_SWZ4(zwww)

        EXPR_IVEC_SWZ4(wxxx) EXPR_IVEC_SWZ4(wxxy) EXPR_IVEC_SWZ4(wxxz) EXPR_IVEC_SWZ4(wxxw) EXPR_IVEC_SWZ4(wxyx) EXPR_IVEC_SWZ4(wxyy) EXPR_IVEC_SWZ4(wxyz) EXPR_IVEC_SWZ4(wxyw)
        EXPR_IVEC_SWZ4(wxzx) EXPR_IVEC_SWZ4(wxzy) EXPR_IVEC_SWZ4(wxzz) EXPR_IVEC_SWZ4(wxzw) EXPR_IVEC_SWZ4(wxwx) EXPR_IVEC_SWZ4(wxwy) EXPR_IVEC_SWZ4(wxwz) EXPR_IVEC_SWZ4(wxww)

        EXPR_IVEC_SWZ4(wyxx) EXPR_IVEC_SWZ4(wyxy) EXPR_IVEC_SWZ4(wyxz) EXPR_IVEC_SWZ4(wyxw) EXPR_IVEC_SWZ4(wyyx) EXPR_IVEC_SWZ4(wyyy) EXPR_IVEC_SWZ4(wyyz) EXPR_IVEC_SWZ4(wyyw)
        EXPR_IVEC_SWZ4(wyzx) EXPR_IVEC_SWZ4(wyzy) EXPR_IVEC_SWZ4(wyzz) EXPR_IVEC_SWZ4(wyzw) EXPR_IVEC_SWZ4(wywx) EXPR_IVEC_SWZ4(wywy) EXPR_IVEC_SWZ4(wywz) EXPR_IVEC_SWZ4(wyww)

        EXPR_IVEC_SWZ4(wzxx) EXPR_IVEC_SWZ4(wzxy) EXPR_IVEC_SWZ4(wzxz) EXPR_IVEC_SWZ4(wzxw) EXPR_IVEC_SWZ4(wzyx) EXPR_IVEC_SWZ4(wzyy) EXPR_IVEC_SWZ4(wzyz) EXPR_IVEC_SWZ4(wzyw)
        EXPR_IVEC_SWZ4(wzzx) EXPR_IVEC_SWZ4(wzzy) EXPR_IVEC_SWZ4(wzzz) EXPR_IVEC_SWZ4(wzzw) EXPR_IVEC_SWZ4(wzwx) EXPR_IVEC_SWZ4(wzwy) EXPR_IVEC_SWZ4(wzwz) EXPR_IVEC_SWZ4(wzww)

        EXPR_IVEC_SWZ4(wwxx) EXPR_IVEC_SWZ4(wwxy) EXPR_IVEC_SWZ4(wwxz) EXPR_IVEC_SWZ4(wwxw) EXPR_IVEC_SWZ4(wwyx) EXPR_IVEC_SWZ4(wwyy) EXPR_IVEC_SWZ4(wwyz) EXPR_IVEC_SWZ4(wwyw)
        EXPR_IVEC_SWZ4(wwzx) EXPR_IVEC_SWZ4(wwzy) EXPR_IVEC_SWZ4(wwzz) EXPR_IVEC_SWZ4(wwzw) EXPR_IVEC_SWZ4(wwwx) EXPR_IVEC_SWZ4(wwwy) EXPR_IVEC_SWZ4(wwwz) EXPR_IVEC_SWZ4(wwww)
        /* clang-format on */

    public:
        // Arithmetic operations
        friend Expr<Math::IVec4> operator+(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator-(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator*(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator/(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Scalar operations with int literal
        friend Expr<Math::IVec4> operator*(Expr<Math::IVec4> lhs, int rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec4> operator*(int lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec4> operator/(Expr<Math::IVec4> lhs, int rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec4> operator+(Expr<Math::IVec4> lhs, int rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec4> operator+(int lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }
        friend Expr<Math::IVec4> operator-(Expr<Math::IVec4> lhs, int rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }
        friend Expr<Math::IVec4> operator-(int lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Scalar operations with Var<int>
        friend Expr<Math::IVec4> operator*(Expr<Math::IVec4> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec4> operator*(const Var<int>& lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec4> operator/(Expr<Math::IVec4> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec4> operator+(Expr<Math::IVec4> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec4> operator+(const Var<int>& lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::IVec4> operator-(Expr<Math::IVec4> lhs, const Var<int>& rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::IVec4> operator-(const Var<int>& lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Load(), rhs.Release()));
        }

        // Bitwise operations
        friend Expr<Math::IVec4> operator&(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator|(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator^(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator~(Expr<Math::IVec4> val) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitNot, val.Release()));
        }

        friend Expr<Math::IVec4> operator<<(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shl, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::IVec4> operator>>(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<Math::IVec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shr, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::IVec4> lhs, Expr<Math::IVec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }
    };

#undef EXPR_IVEC_MEM
#undef EXPR_IVEC_SWZ2
#undef EXPR_IVEC_SWZ3
#undef EXPR_IVEC_SWZ4

    // ==================== Var IVector Scalar Operations ====================
    // IVec2 * int
    [[nodiscard]] inline Expr<Math::IVec2> operator*(const VarBase<Math::IVec2> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator*(int lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 / int
    [[nodiscard]] inline Expr<Math::IVec2> operator/(const VarBase<Math::IVec2> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 + int
    [[nodiscard]] inline Expr<Math::IVec2> operator+(const VarBase<Math::IVec2> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator+(int lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 - int
    [[nodiscard]] inline Expr<Math::IVec2> operator-(const VarBase<Math::IVec2> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator-(int lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec3 * int
    [[nodiscard]] inline Expr<Math::IVec3> operator*(const VarBase<Math::IVec3> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator*(int lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 / int
    [[nodiscard]] inline Expr<Math::IVec3> operator/(const VarBase<Math::IVec3> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 + int
    [[nodiscard]] inline Expr<Math::IVec3> operator+(const VarBase<Math::IVec3> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator+(int lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 - int
    [[nodiscard]] inline Expr<Math::IVec3> operator-(const VarBase<Math::IVec3> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator-(int lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec4 * int
    [[nodiscard]] inline Expr<Math::IVec4> operator*(const VarBase<Math::IVec4> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator*(int lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 / int
    [[nodiscard]] inline Expr<Math::IVec4> operator/(const VarBase<Math::IVec4> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 + int
    [[nodiscard]] inline Expr<Math::IVec4> operator+(const VarBase<Math::IVec4> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator+(int lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 - int
    [[nodiscard]] inline Expr<Math::IVec4> operator-(const VarBase<Math::IVec4> &lhs, int rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator-(int lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // ==================== Var IVector with Var<int> Operations ====================
    // IVec2 * Var<int>
    [[nodiscard]] inline Expr<Math::IVec2> operator*(const VarBase<Math::IVec2> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator*(const Var<int> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 / Var<int>
    [[nodiscard]] inline Expr<Math::IVec2> operator/(const VarBase<Math::IVec2> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 + Var<int>
    [[nodiscard]] inline Expr<Math::IVec2> operator+(const VarBase<Math::IVec2> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator+(const Var<int> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec2 - Var<int>
    [[nodiscard]] inline Expr<Math::IVec2> operator-(const VarBase<Math::IVec2> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator-(const Var<int> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec3 * Var<int>
    [[nodiscard]] inline Expr<Math::IVec3> operator*(const VarBase<Math::IVec3> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator*(const Var<int> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 / Var<int>
    [[nodiscard]] inline Expr<Math::IVec3> operator/(const VarBase<Math::IVec3> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 + Var<int>
    [[nodiscard]] inline Expr<Math::IVec3> operator+(const VarBase<Math::IVec3> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator+(const Var<int> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec3 - Var<int>
    [[nodiscard]] inline Expr<Math::IVec3> operator-(const VarBase<Math::IVec3> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator-(const Var<int> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec4 * Var<int>
    [[nodiscard]] inline Expr<Math::IVec4> operator*(const VarBase<Math::IVec4> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator*(const Var<int> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 / Var<int>
    [[nodiscard]] inline Expr<Math::IVec4> operator/(const VarBase<Math::IVec4> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 + Var<int>
    [[nodiscard]] inline Expr<Math::IVec4> operator+(const VarBase<Math::IVec4> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator+(const Var<int> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // IVec4 - Var<int>
    [[nodiscard]] inline Expr<Math::IVec4> operator-(const VarBase<Math::IVec4> &lhs, const Var<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator-(const Var<int> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec2 VarBase & VarBase operations
    [[nodiscard]] inline Expr<Math::IVec2> operator&(const VarBase<Math::IVec2> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator|(const VarBase<Math::IVec2> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator^(const VarBase<Math::IVec2> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator<<(const VarBase<Math::IVec2> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec2> operator>>(const VarBase<Math::IVec2> &lhs, const VarBase<Math::IVec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec3 VarBase & VarBase operations
    [[nodiscard]] inline Expr<Math::IVec3> operator&(const VarBase<Math::IVec3> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator|(const VarBase<Math::IVec3> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator^(const VarBase<Math::IVec3> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator<<(const VarBase<Math::IVec3> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec3> operator>>(const VarBase<Math::IVec3> &lhs, const VarBase<Math::IVec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // IVec4 VarBase & VarBase operations
    [[nodiscard]] inline Expr<Math::IVec4> operator&(const VarBase<Math::IVec4> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator|(const VarBase<Math::IVec4> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator^(const VarBase<Math::IVec4> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator<<(const VarBase<Math::IVec4> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::IVec4> operator>>(const VarBase<Math::IVec4> &lhs, const VarBase<Math::IVec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::IVec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

}

#endif //EASYGPU_EXPRIVECTOR_H
