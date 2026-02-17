#pragma once

/**
 * ExprVector.h:
 *      @Descripiton    :   The specified expression API for vector with swizzle access
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_EXPRVECTOR_H
#define EASYGPU_EXPRVECTOR_H

#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <Utility/Vec.h>

#include <format>

namespace GPU::IR::Value {
    // Forward declaration of Var for friend declarations
    template<ScalarType T>
    class Var;

    // Swizzle macros for expression vectors
    // Single component access returns Expr<float>
#define EXPR_MEM(n) Expr<float> n() { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<float>(std::make_unique<Node::LoadUniformNode>(std::format("({}).{}", exprStr, #n))); \
}

    // 2-component swizzle
#define EXPR_SWZ2(n) Expr<Math::Vec2> n() { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::Vec2>(std::make_unique<Node::LoadUniformNode>(std::format("({}).{}", exprStr, #n))); \
}

    // 3-component swizzle
#define EXPR_SWZ3(n) Expr<Math::Vec3> n() { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::Vec3>(std::make_unique<Node::LoadUniformNode>(std::format("({}).{}", exprStr, #n))); \
}

    // 4-component swizzle
#define EXPR_SWZ4(n) Expr<Math::Vec4> n() { \
    std::string exprStr = Builder::Builder::Get().BuildNode(*this->Node()); \
    return Expr<Math::Vec4>(std::make_unique<Node::LoadUniformNode>(std::format("({}).{}", exprStr, #n))); \
}

    // Specialization for Vec2 expressions with swizzle access
    template<>
    class Expr<Math::Vec2> : public ExprBase {
    public:
        using ValueType = Math::Vec2;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        explicit Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        explicit Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::Vec2>& var);
        
        // Construct from components
        template<typename X, typename Y>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>)
        Expr(X&& x, Y&& y) {
            std::string xStr = ToGLSLString(std::forward<X>(x));
            std::string yStr = ToGLSLString(std::forward<Y>(y));
            _node = std::make_unique<Node::LoadUniformNode>(std::format("vec2({}, {})", xStr, yStr));
        }
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) & = delete;

        Expr<float> operator[](ExprBase index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<float> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_MEM(x) EXPR_MEM(y)

        // 2-component swizzles
        EXPR_SWZ2(xx) EXPR_SWZ2(xy) EXPR_SWZ2(yx) EXPR_SWZ2(yy)
        /* clang-format on */

    private:
        // Helper to convert value to GLSL string
        template<typename T>
        static std::string ToGLSLString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, float>) {
                return ValueToString(val);
            } else if constexpr (std::same_as<U, Expr<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }

    public:
        // Arithmetic operations
        friend Expr<Math::Vec2> operator+(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec2> operator-(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec2> operator*(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec2> operator/(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::Vec2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for Vec3 expressions with swizzle access
    template<>
    class Expr<Math::Vec3> : public ExprBase {
    public:
        using ValueType = Math::Vec3;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        explicit Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        explicit Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::Vec3>& var);
        
        // Construct from components
        template<typename X, typename Y, typename Z>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>) &&
                     (std::same_as<std::remove_cvref_t<Z>, Var<float>> || std::same_as<std::remove_cvref_t<Z>, Expr<float>> || std::same_as<std::remove_cvref_t<Z>, float>)
        Expr(X&& x, Y&& y, Z&& z) {
            std::string xStr = ToGLSLString(std::forward<X>(x));
            std::string yStr = ToGLSLString(std::forward<Y>(y));
            std::string zStr = ToGLSLString(std::forward<Z>(z));
            _node = std::make_unique<Node::LoadUniformNode>(std::format("vec3({}, {}, {})", xStr, yStr, zStr));
        }
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) & = delete;

        Expr<float> operator[](ExprBase index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<float> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_MEM(x) EXPR_MEM(y) EXPR_MEM(z)

        // 2-component swizzles
        EXPR_SWZ2(xx) EXPR_SWZ2(xy) EXPR_SWZ2(xz)
        EXPR_SWZ2(yx) EXPR_SWZ2(yy) EXPR_SWZ2(yz)
        EXPR_SWZ2(zx) EXPR_SWZ2(zy) EXPR_SWZ2(zz)

        // 3-component swizzles
        EXPR_SWZ3(xxx) EXPR_SWZ3(xxy) EXPR_SWZ3(xxz)
        EXPR_SWZ3(xyx) EXPR_SWZ3(xyy) EXPR_SWZ3(xyz)
        EXPR_SWZ3(xzx) EXPR_SWZ3(xzy) EXPR_SWZ3(xzz)

        EXPR_SWZ3(yxx) EXPR_SWZ3(yxy) EXPR_SWZ3(yxz)
        EXPR_SWZ3(yyx) EXPR_SWZ3(yyy) EXPR_SWZ3(yyz)
        EXPR_SWZ3(yzx) EXPR_SWZ3(yzy) EXPR_SWZ3(yzz)

        EXPR_SWZ3(zxx) EXPR_SWZ3(zxy) EXPR_SWZ3(zxz)
        EXPR_SWZ3(zyx) EXPR_SWZ3(zyy) EXPR_SWZ3(zyz)
        EXPR_SWZ3(zzx) EXPR_SWZ3(zzy) EXPR_SWZ3(zzz)
        /* clang-format on */

    private:
        // Helper to convert value to GLSL string
        template<typename T>
        static std::string ToGLSLString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, float>) {
                return ValueToString(val);
            } else if constexpr (std::same_as<U, Expr<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }

    public:
        // Arithmetic operations
        friend Expr<Math::Vec3> operator+(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec3> operator-(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec3> operator*(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec3> operator/(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::Vec3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for Vec4 expressions with swizzle access
    template<>
    class Expr<Math::Vec4> : public ExprBase {
    public:
        using ValueType = Math::Vec4;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        explicit Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        explicit Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Construct from same-type Var
        Expr(const Var<Math::Vec4>& var);
        
        // Construct from components
        template<typename X, typename Y, typename Z, typename W>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>) &&
                     (std::same_as<std::remove_cvref_t<Z>, Var<float>> || std::same_as<std::remove_cvref_t<Z>, Expr<float>> || std::same_as<std::remove_cvref_t<Z>, float>) &&
                     (std::same_as<std::remove_cvref_t<W>, Var<float>> || std::same_as<std::remove_cvref_t<W>, Expr<float>> || std::same_as<std::remove_cvref_t<W>, float>)
        Expr(X&& x, Y&& y, Z&& z, W&& w) {
            std::string xStr = ToGLSLString(std::forward<X>(x));
            std::string yStr = ToGLSLString(std::forward<Y>(y));
            std::string zStr = ToGLSLString(std::forward<Z>(z));
            std::string wStr = ToGLSLString(std::forward<W>(w));
            _node = std::make_unique<Node::LoadUniformNode>(std::format("vec4({}, {}, {}, {})", xStr, yStr, zStr, wStr));
        }
        
        ~Expr() = default;

        // Subscript access
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<float> operator[](IndexType index) & = delete;

        Expr<float> operator[](ExprBase index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<float> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) && {
            return Expr<float>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<float> operator[](Expr<IndexT> index) & = delete;

        // Swizzle access (rvalue only)
        /* clang-format off */
        EXPR_MEM(x) EXPR_MEM(y) EXPR_MEM(z) EXPR_MEM(w)

        // 2-component swizzles (16)
        EXPR_SWZ2(xx) EXPR_SWZ2(xy) EXPR_SWZ2(xz) EXPR_SWZ2(xw)
        EXPR_SWZ2(yx) EXPR_SWZ2(yy) EXPR_SWZ2(yz) EXPR_SWZ2(yw)
        EXPR_SWZ2(zx) EXPR_SWZ2(zy) EXPR_SWZ2(zz) EXPR_SWZ2(zw)
        EXPR_SWZ2(wx) EXPR_SWZ2(wy) EXPR_SWZ2(wz) EXPR_SWZ2(ww)

        // 3-component swizzles (64)
        EXPR_SWZ3(xxx) EXPR_SWZ3(xxy) EXPR_SWZ3(xxz) EXPR_SWZ3(xxw) EXPR_SWZ3(xyx) EXPR_SWZ3(xyy) EXPR_SWZ3(xyz) EXPR_SWZ3(xyw)
        EXPR_SWZ3(xzx) EXPR_SWZ3(xzy) EXPR_SWZ3(xzz) EXPR_SWZ3(xzw) EXPR_SWZ3(xwx) EXPR_SWZ3(xwy) EXPR_SWZ3(xwz) EXPR_SWZ3(xww)

        EXPR_SWZ3(yxx) EXPR_SWZ3(yxy) EXPR_SWZ3(yxz) EXPR_SWZ3(yxw) EXPR_SWZ3(yyx) EXPR_SWZ3(yyy) EXPR_SWZ3(yyz) EXPR_SWZ3(yyw)
        EXPR_SWZ3(yzx) EXPR_SWZ3(yzy) EXPR_SWZ3(yzz) EXPR_SWZ3(yzw) EXPR_SWZ3(ywx) EXPR_SWZ3(ywy) EXPR_SWZ3(ywz) EXPR_SWZ3(yww)

        EXPR_SWZ3(zxx) EXPR_SWZ3(zxy) EXPR_SWZ3(zxz) EXPR_SWZ3(zxw) EXPR_SWZ3(zyx) EXPR_SWZ3(zyy) EXPR_SWZ3(zyz) EXPR_SWZ3(zyw)
        EXPR_SWZ3(zzx) EXPR_SWZ3(zzy) EXPR_SWZ3(zzz) EXPR_SWZ3(zzw) EXPR_SWZ3(zwx) EXPR_SWZ3(zwy) EXPR_SWZ3(zwz) EXPR_SWZ3(zww)

        EXPR_SWZ3(wxx) EXPR_SWZ3(wxy) EXPR_SWZ3(wxz) EXPR_SWZ3(wxw) EXPR_SWZ3(wyx) EXPR_SWZ3(wyy) EXPR_SWZ3(wyz) EXPR_SWZ3(wyw)
        EXPR_SWZ3(wzx) EXPR_SWZ3(wzy) EXPR_SWZ3(wzz) EXPR_SWZ3(wzw) EXPR_SWZ3(wwx) EXPR_SWZ3(wwy) EXPR_SWZ3(wwz) EXPR_SWZ3(www)

        // 4-component swizzles (256) - Partial, commonly used ones
        EXPR_SWZ4(xxxx) EXPR_SWZ4(xxxy) EXPR_SWZ4(xxxz) EXPR_SWZ4(xxxw) EXPR_SWZ4(xxyx) EXPR_SWZ4(xxyy) EXPR_SWZ4(xxyz) EXPR_SWZ4(xxyw)
        EXPR_SWZ4(xxzx) EXPR_SWZ4(xxzy) EXPR_SWZ4(xxzz) EXPR_SWZ4(xxzw) EXPR_SWZ4(xxwx) EXPR_SWZ4(xxwy) EXPR_SWZ4(xxwz) EXPR_SWZ4(xxww)

        EXPR_SWZ4(xyxx) EXPR_SWZ4(xyxy) EXPR_SWZ4(xyxz) EXPR_SWZ4(xyxw) EXPR_SWZ4(xyyx) EXPR_SWZ4(xyyy) EXPR_SWZ4(xyyz) EXPR_SWZ4(xyyw)
        EXPR_SWZ4(xyzx) EXPR_SWZ4(xyzy) EXPR_SWZ4(xyzz) EXPR_SWZ4(xyzw) EXPR_SWZ4(xywx) EXPR_SWZ4(xywy) EXPR_SWZ4(xywz) EXPR_SWZ4(xyww)

        EXPR_SWZ4(xzxx) EXPR_SWZ4(xzxy) EXPR_SWZ4(xzxz) EXPR_SWZ4(xzxw) EXPR_SWZ4(xzyx) EXPR_SWZ4(xzyy) EXPR_SWZ4(xzyz) EXPR_SWZ4(xzyw)
        EXPR_SWZ4(xzzx) EXPR_SWZ4(xzzy) EXPR_SWZ4(xzzz) EXPR_SWZ4(xzzw) EXPR_SWZ4(xzwx) EXPR_SWZ4(xzwy) EXPR_SWZ4(xzwz) EXPR_SWZ4(xzww)

        EXPR_SWZ4(xwxx) EXPR_SWZ4(xwxy) EXPR_SWZ4(xwxz) EXPR_SWZ4(xwxw) EXPR_SWZ4(xwyx) EXPR_SWZ4(xwyy) EXPR_SWZ4(xwyz) EXPR_SWZ4(xwyw)
        EXPR_SWZ4(xwzx) EXPR_SWZ4(xwzy) EXPR_SWZ4(xwzz) EXPR_SWZ4(xwzw) EXPR_SWZ4(xwwx) EXPR_SWZ4(xwwy) EXPR_SWZ4(xwwz) EXPR_SWZ4(xwww)

        EXPR_SWZ4(yxxx) EXPR_SWZ4(yxxy) EXPR_SWZ4(yxxz) EXPR_SWZ4(yxxw) EXPR_SWZ4(yxyx) EXPR_SWZ4(yxyy) EXPR_SWZ4(yxyz) EXPR_SWZ4(yxyw)
        EXPR_SWZ4(yxzx) EXPR_SWZ4(yxzy) EXPR_SWZ4(yxzz) EXPR_SWZ4(yxzw) EXPR_SWZ4(yxwx) EXPR_SWZ4(yxwy) EXPR_SWZ4(yxwz) EXPR_SWZ4(yxww)

        EXPR_SWZ4(yyxx) EXPR_SWZ4(yyxy) EXPR_SWZ4(yyxz) EXPR_SWZ4(yyxw) EXPR_SWZ4(yyyx) EXPR_SWZ4(yyyy) EXPR_SWZ4(yyyz) EXPR_SWZ4(yyyw)
        EXPR_SWZ4(yyzx) EXPR_SWZ4(yyzy) EXPR_SWZ4(yyzz) EXPR_SWZ4(yyzw) EXPR_SWZ4(yywx) EXPR_SWZ4(yywy) EXPR_SWZ4(yywz) EXPR_SWZ4(yyww)

        EXPR_SWZ4(yzxx) EXPR_SWZ4(yzxy) EXPR_SWZ4(yzxz) EXPR_SWZ4(yzxw) EXPR_SWZ4(yzyx) EXPR_SWZ4(yzyy) EXPR_SWZ4(yzyz) EXPR_SWZ4(yzyw)
        EXPR_SWZ4(yzzx) EXPR_SWZ4(yzzy) EXPR_SWZ4(yzzz) EXPR_SWZ4(yzzw) EXPR_SWZ4(yzwx) EXPR_SWZ4(yzwy) EXPR_SWZ4(yzwz) EXPR_SWZ4(yzww)

        EXPR_SWZ4(ywxx) EXPR_SWZ4(ywxy) EXPR_SWZ4(ywxz) EXPR_SWZ4(ywxw) EXPR_SWZ4(ywyx) EXPR_SWZ4(ywyy) EXPR_SWZ4(ywyz) EXPR_SWZ4(ywyw)
        EXPR_SWZ4(ywzx) EXPR_SWZ4(ywzy) EXPR_SWZ4(ywzz) EXPR_SWZ4(ywzw) EXPR_SWZ4(ywwx) EXPR_SWZ4(ywwy) EXPR_SWZ4(ywwz) EXPR_SWZ4(ywww)

        EXPR_SWZ4(zxxx) EXPR_SWZ4(zxxy) EXPR_SWZ4(zxxz) EXPR_SWZ4(zxxw) EXPR_SWZ4(zxyx) EXPR_SWZ4(zxyy) EXPR_SWZ4(zxyz) EXPR_SWZ4(zxyw)
        EXPR_SWZ4(zxzx) EXPR_SWZ4(zxzy) EXPR_SWZ4(zxzz) EXPR_SWZ4(zxzw) EXPR_SWZ4(zxwx) EXPR_SWZ4(zxwy) EXPR_SWZ4(zxwz) EXPR_SWZ4(zxww)

        EXPR_SWZ4(zyxx) EXPR_SWZ4(zyxy) EXPR_SWZ4(zyxz) EXPR_SWZ4(zyxw) EXPR_SWZ4(zyyx) EXPR_SWZ4(zyyy) EXPR_SWZ4(zyyz) EXPR_SWZ4(zyyw)
        EXPR_SWZ4(zyzx) EXPR_SWZ4(zyzy) EXPR_SWZ4(zyzz) EXPR_SWZ4(zyzw) EXPR_SWZ4(zywx) EXPR_SWZ4(zywy) EXPR_SWZ4(zywz) EXPR_SWZ4(zyww)

        EXPR_SWZ4(zzxx) EXPR_SWZ4(zzxy) EXPR_SWZ4(zzxz) EXPR_SWZ4(zzxw) EXPR_SWZ4(zzyx) EXPR_SWZ4(zzyy) EXPR_SWZ4(zzyz) EXPR_SWZ4(zzyw)
        EXPR_SWZ4(zzzx) EXPR_SWZ4(zzzy) EXPR_SWZ4(zzzz) EXPR_SWZ4(zzzw) EXPR_SWZ4(zzwx) EXPR_SWZ4(zzwy) EXPR_SWZ4(zzwz) EXPR_SWZ4(zzww)

        EXPR_SWZ4(zwxx) EXPR_SWZ4(zwxy) EXPR_SWZ4(zwxz) EXPR_SWZ4(zwxw) EXPR_SWZ4(zwyx) EXPR_SWZ4(zwyy) EXPR_SWZ4(zwyz) EXPR_SWZ4(zwyw)
        EXPR_SWZ4(zwzx) EXPR_SWZ4(zwzy) EXPR_SWZ4(zwzz) EXPR_SWZ4(zwzw) EXPR_SWZ4(zwwx) EXPR_SWZ4(zwwy) EXPR_SWZ4(zwwz) EXPR_SWZ4(zwww)

        EXPR_SWZ4(wxxx) EXPR_SWZ4(wxxy) EXPR_SWZ4(wxxz) EXPR_SWZ4(wxxw) EXPR_SWZ4(wxyx) EXPR_SWZ4(wxyy) EXPR_SWZ4(wxyz) EXPR_SWZ4(wxyw)
        EXPR_SWZ4(wxzx) EXPR_SWZ4(wxzy) EXPR_SWZ4(wxzz) EXPR_SWZ4(wxzw) EXPR_SWZ4(wxwx) EXPR_SWZ4(wxwy) EXPR_SWZ4(wxwz) EXPR_SWZ4(wxww)

        EXPR_SWZ4(wyxx) EXPR_SWZ4(wyxy) EXPR_SWZ4(wyxz) EXPR_SWZ4(wyxw) EXPR_SWZ4(wyyx) EXPR_SWZ4(wyyy) EXPR_SWZ4(wyyz) EXPR_SWZ4(wyyw)
        EXPR_SWZ4(wyzx) EXPR_SWZ4(wyzy) EXPR_SWZ4(wyzz) EXPR_SWZ4(wyzw) EXPR_SWZ4(wywx) EXPR_SWZ4(wywy) EXPR_SWZ4(wywz) EXPR_SWZ4(wyww)

        EXPR_SWZ4(wzxx) EXPR_SWZ4(wzxy) EXPR_SWZ4(wzxz) EXPR_SWZ4(wzxw) EXPR_SWZ4(wzyx) EXPR_SWZ4(wzyy) EXPR_SWZ4(wzyz) EXPR_SWZ4(wzyw)
        EXPR_SWZ4(wzzx) EXPR_SWZ4(wzzy) EXPR_SWZ4(wzzz) EXPR_SWZ4(wzzw) EXPR_SWZ4(wzwx) EXPR_SWZ4(wzwy) EXPR_SWZ4(wzwz) EXPR_SWZ4(wzww)

        EXPR_SWZ4(wwxx) EXPR_SWZ4(wwxy) EXPR_SWZ4(wwxz) EXPR_SWZ4(wwxw) EXPR_SWZ4(wwyx) EXPR_SWZ4(wwyy) EXPR_SWZ4(wwyz) EXPR_SWZ4(wwyw)
        EXPR_SWZ4(wwzx) EXPR_SWZ4(wwzy) EXPR_SWZ4(wwzz) EXPR_SWZ4(wwzw) EXPR_SWZ4(wwwx) EXPR_SWZ4(wwwy) EXPR_SWZ4(wwwz) EXPR_SWZ4(wwww)
        /* clang-format on */

    public:
        // Arithmetic operations
        friend Expr<Math::Vec4> operator+(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec4> operator-(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec4> operator*(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Vec4> operator/(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Release()));
        }

        // Comparison
        friend Expr<bool> operator<(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<Math::Vec4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }

    private:
        // Helper to convert value to GLSL string
        template<typename T>
        static std::string ToGLSLString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, float>) {
                return ValueToString(val);
            } else if constexpr (std::same_as<U, Expr<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<float>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

#undef EXPR_MEM
#undef EXPR_SWZ2
#undef EXPR_SWZ3
#undef EXPR_SWZ4


}

#endif //EASYGPU_EXPRVECTOR_H
