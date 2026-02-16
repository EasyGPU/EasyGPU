#pragma once

/**
 * VarVector.h:
 *      @Descripiton    :   The specified variable API for vector
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_VARVECTOR_H
#define EASYGPU_VARVECTOR_H

// Note: This file should only be included by Var.h after Var main template is defined
// Do not include Var.h here to avoid circular inclusion
#include <IR/Value/Expr.h>
#include <IR/Value/ExprVector.h>

#include <Utility/Vec.h>

#include <format>

namespace GPU::IR::Value {
    // Swizzle macros for vectors
#define MEM(n) Var<float> n() { return std::move(Var<float>(std::format("{}.{}", _varNode->VarName(), #n))); }

#define SWZ2(n) Var<Math::Vec2> n() { return std::move(Var<Math::Vec2>(std::format("{}.{}", _varNode->VarName(), #n))); }
#define SWZ3(n) Var<Math::Vec3> n() { return std::move(Var<Math::Vec3>(std::format("{}.{}", _varNode->VarName(), #n))); }
#define SWZ4(n) Var<Math::Vec4> n() { return std::move(Var<Math::Vec4>(std::format("{}.{}", _varNode->VarName(), #n))); }

    // Swizzle accessing for Vec2
    template<>
    class Var<Math::Vec2> : public VarBase<Math::Vec2> {
    public:
        using VarBase<Math::Vec2>::VarBase;
        using VarBase<Math::Vec2>::Load;
        using VarBase<Math::Vec2>::operator=;
        using VarBase<Math::Vec2>::operator Expr<Math::Vec2>;

        // Component constructor: Var<Vec2> v(x, y)
        template<typename X, typename Y>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>)
        Var(X&& x, Y&& y) : VarBase() {
            std::string xStr = ComponentToString(std::forward<X>(x));
            std::string yStr = ComponentToString(std::forward<Y>(y));
            auto initCode = std::format("{}=vec2({}, {});\n", _varNode->VarName(), xStr, yStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<float> operator[](T Index) {
            return Var<float>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<float> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<float> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    public:
        /* clang-format off */
        MEM(x) MEM(y)

        // 2-component swizzles (4)
        SWZ2(xx) SWZ2(xy) SWZ2(yx) SWZ2(yy)
        /* clang-format on */

    private:
        // Helper to convert component to string
        template<typename T>
        static std::string ComponentToString(T&& val) {
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

    // Swizzle accessing for Vec3
    template<>
    class Var<Math::Vec3> : public VarBase<Math::Vec3> {
    public:
        using VarBase<Math::Vec3>::VarBase;
        using VarBase<Math::Vec3>::Load;
        using VarBase<Math::Vec3>::operator=;
        using VarBase<Math::Vec3>::operator Expr<Math::Vec3>;

        // Component constructor: Var<Vec3> v(x, y, z)
        template<typename X, typename Y, typename Z>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>) &&
                     (std::same_as<std::remove_cvref_t<Z>, Var<float>> || std::same_as<std::remove_cvref_t<Z>, Expr<float>> || std::same_as<std::remove_cvref_t<Z>, float>)
        Var(X&& x, Y&& y, Z&& z) : VarBase() {
            std::string xStr = ComponentToString(std::forward<X>(x));
            std::string yStr = ComponentToString(std::forward<Y>(y));
            std::string zStr = ComponentToString(std::forward<Z>(z));
            auto initCode = std::format("{}=vec3({}, {}, {});\n", _varNode->VarName(), xStr, yStr, zStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<float> operator[](T Index) {
            return Var<float>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<float> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<float> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    public:
        /* clang-format off */
        MEM(x) MEM(y) MEM(z)

        // 2-component swizzles
        SWZ2(xx) SWZ2(xy) SWZ2(xz)
        SWZ2(yx) SWZ2(yy) SWZ2(yz)
        SWZ2(zx) SWZ2(zy) SWZ2(zz)

        // 3-component swizzles
        SWZ3(xxx) SWZ3(xxy) SWZ3(xxz)
        SWZ3(xyx) SWZ3(xyy) SWZ3(xyz)
        SWZ3(xzx) SWZ3(xzy) SWZ3(xzz)

        SWZ3(yxx) SWZ3(yxy) SWZ3(yxz)
        SWZ3(yyx) SWZ3(yyy) SWZ3(yyz)
        SWZ3(yzx) SWZ3(yzy) SWZ3(yzz)

        SWZ3(zxx) SWZ3(zxy) SWZ3(zxz)
        SWZ3(zyx) SWZ3(zyy) SWZ3(zyz)
        SWZ3(zzx) SWZ3(zzy) SWZ3(zzz)
        /* clang-format on */

    private:
        // Helper to convert component to string
        template<typename T>
        static std::string ComponentToString(T&& val) {
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

    // Swizzle accessing for Vec4

    template<>
    class Var<Math::Vec4> : public VarBase<Math::Vec4> {
    public:
        using VarBase<Math::Vec4>::VarBase;
        using VarBase<Math::Vec4>::Load;
        using VarBase<Math::Vec4>::operator=;
        using VarBase<Math::Vec4>::operator Expr<Math::Vec4>;

        // Component constructor: Var<Vec4> v(x, y, z, w)
        template<typename X, typename Y, typename Z, typename W>
            requires (std::same_as<std::remove_cvref_t<X>, Var<float>> || std::same_as<std::remove_cvref_t<X>, Expr<float>> || std::same_as<std::remove_cvref_t<X>, float>) &&
                     (std::same_as<std::remove_cvref_t<Y>, Var<float>> || std::same_as<std::remove_cvref_t<Y>, Expr<float>> || std::same_as<std::remove_cvref_t<Y>, float>) &&
                     (std::same_as<std::remove_cvref_t<Z>, Var<float>> || std::same_as<std::remove_cvref_t<Z>, Expr<float>> || std::same_as<std::remove_cvref_t<Z>, float>) &&
                     (std::same_as<std::remove_cvref_t<W>, Var<float>> || std::same_as<std::remove_cvref_t<W>, Expr<float>> || std::same_as<std::remove_cvref_t<W>, float>)
        Var(X&& x, Y&& y, Z&& z, W&& w) : VarBase() {
            std::string xStr = ComponentToString(std::forward<X>(x));
            std::string yStr = ComponentToString(std::forward<Y>(y));
            std::string zStr = ComponentToString(std::forward<Z>(z));
            std::string wStr = ComponentToString(std::forward<W>(w));
            auto initCode = std::format("{}=vec4({}, {}, {}, {});\n", _varNode->VarName(), xStr, yStr, zStr, wStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<float> operator[](T Index) {
            return Var<float>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<float> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<float> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    public:
        /* clang-format off */
        MEM(x) MEM(y) MEM(z) MEM(w)

        // 2-component swizzles (16)
        SWZ2(xx) SWZ2(xy) SWZ2(xz) SWZ2(xw)
        SWZ2(yx) SWZ2(yy) SWZ2(yz) SWZ2(yw)
        SWZ2(zx) SWZ2(zy) SWZ2(zz) SWZ2(zw)
        SWZ2(wx) SWZ2(wy) SWZ2(wz) SWZ2(ww)

        // 3-component swizzles (64)
        SWZ3(xxx) SWZ3(xxy) SWZ3(xxz) SWZ3(xxw) SWZ3(xyx) SWZ3(xyy) SWZ3(xyz) SWZ3(xyw)
        SWZ3(xzx) SWZ3(xzy) SWZ3(xzz) SWZ3(xzw) SWZ3(xwx) SWZ3(xwy) SWZ3(xwz) SWZ3(xww)

        SWZ3(yxx) SWZ3(yxy) SWZ3(yxz) SWZ3(yxw) SWZ3(yyx) SWZ3(yyy) SWZ3(yyz) SWZ3(yyw)
        SWZ3(yzx) SWZ3(yzy) SWZ3(yzz) SWZ3(yzw) SWZ3(ywx) SWZ3(ywy) SWZ3(ywz) SWZ3(yww)

        SWZ3(zxx) SWZ3(zxy) SWZ3(zxz) SWZ3(zxw) SWZ3(zyx) SWZ3(zyy) SWZ3(zyz) SWZ3(zyw)
        SWZ3(zzx) SWZ3(zzy) SWZ3(zzz) SWZ3(zzw) SWZ3(zwx) SWZ3(zwy) SWZ3(zwz) SWZ3(zww)

        SWZ3(wxx) SWZ3(wxy) SWZ3(wxz) SWZ3(wxw) SWZ3(wyx) SWZ3(wyy) SWZ3(wyz) SWZ3(wyw)
        SWZ3(wzx) SWZ3(wzy) SWZ3(wzz) SWZ3(wzw) SWZ3(wwx) SWZ3(wwy) SWZ3(wwz) SWZ3(www)

        // 4-component swizzles (256)
        SWZ4(xxxx) SWZ4(xxxy) SWZ4(xxxz) SWZ4(xxxw) SWZ4(xxyx) SWZ4(xxyy) SWZ4(xxyz) SWZ4(xxyw)
        SWZ4(xxzx) SWZ4(xxzy) SWZ4(xxzz) SWZ4(xxzw) SWZ4(xxwx) SWZ4(xxwy) SWZ4(xxwz) SWZ4(xxww)

        SWZ4(xyxx) SWZ4(xyxy) SWZ4(xyxz) SWZ4(xyxw) SWZ4(xyyx) SWZ4(xyyy) SWZ4(xyyz) SWZ4(xyyw)
        SWZ4(xyzx) SWZ4(xyzy) SWZ4(xyzz) SWZ4(xyzw) SWZ4(xywx) SWZ4(xywy) SWZ4(xywz) SWZ4(xyww)

        SWZ4(xzxx) SWZ4(xzxy) SWZ4(xzxz) SWZ4(xzxw) SWZ4(xzyx) SWZ4(xzyy) SWZ4(xzyz) SWZ4(xzyw)
        SWZ4(xzzx) SWZ4(xzzy) SWZ4(xzzz) SWZ4(xzzw) SWZ4(xzwx) SWZ4(xzwy) SWZ4(xzwz) SWZ4(xzww)

        SWZ4(xwxx) SWZ4(xwxy) SWZ4(xwxz) SWZ4(xwxw) SWZ4(xwyx) SWZ4(xwyy) SWZ4(xwyz) SWZ4(xwyw)
        SWZ4(xwzx) SWZ4(xwzy) SWZ4(xwzz) SWZ4(xwzw) SWZ4(xwwx) SWZ4(xwwy) SWZ4(xwwz) SWZ4(xwww)

        SWZ4(yxxx) SWZ4(yxxy) SWZ4(yxxz) SWZ4(yxxw) SWZ4(yxyx) SWZ4(yxyy) SWZ4(yxyz) SWZ4(yxyw)
        SWZ4(yxzx) SWZ4(yxzy) SWZ4(yxzz) SWZ4(yxzw) SWZ4(yxwx) SWZ4(yxwy) SWZ4(yxwz) SWZ4(yxww)

        SWZ4(yyxx) SWZ4(yyxy) SWZ4(yyxz) SWZ4(yyxw) SWZ4(yyyx) SWZ4(yyyy) SWZ4(yyyz) SWZ4(yyyw)
        SWZ4(yyzx) SWZ4(yyzy) SWZ4(yyzz) SWZ4(yyzw) SWZ4(yywx) SWZ4(yywy) SWZ4(yywz) SWZ4(yyww)

        SWZ4(yzxx) SWZ4(yzxy) SWZ4(yzxz) SWZ4(yzxw) SWZ4(yzyx) SWZ4(yzyy) SWZ4(yzyz) SWZ4(yzyw)
        SWZ4(yzzx) SWZ4(yzzy) SWZ4(yzzz) SWZ4(yzzw) SWZ4(yzwx) SWZ4(yzwy) SWZ4(yzwz) SWZ4(yzww)

        SWZ4(ywxx) SWZ4(ywxy) SWZ4(ywxz) SWZ4(ywxw) SWZ4(ywyx) SWZ4(ywyy) SWZ4(ywyz) SWZ4(ywyw)
        SWZ4(ywzx) SWZ4(ywzy) SWZ4(ywzz) SWZ4(ywzw) SWZ4(ywwx) SWZ4(ywwy) SWZ4(ywwz) SWZ4(ywww)

        SWZ4(zxxx) SWZ4(zxxy) SWZ4(zxxz) SWZ4(zxxw) SWZ4(zxyx) SWZ4(zxyy) SWZ4(zxyz) SWZ4(zxyw)
        SWZ4(zxzx) SWZ4(zxzy) SWZ4(zxzz) SWZ4(zxzw) SWZ4(zxwx) SWZ4(zxwy) SWZ4(zxwz) SWZ4(zxww)

        SWZ4(zyxx) SWZ4(zyxy) SWZ4(zyxz) SWZ4(zyxw) SWZ4(zyyx) SWZ4(zyyy) SWZ4(zyyz) SWZ4(zyyw)
        SWZ4(zyzx) SWZ4(zyzy) SWZ4(zyzz) SWZ4(zyzw) SWZ4(zywx) SWZ4(zywy) SWZ4(zywz) SWZ4(zyww)

        SWZ4(zzxx) SWZ4(zzxy) SWZ4(zzxz) SWZ4(zzxw) SWZ4(zzyx) SWZ4(zzyy) SWZ4(zzyz) SWZ4(zzyw)
        SWZ4(zzzx) SWZ4(zzzy) SWZ4(zzzz) SWZ4(zzzw) SWZ4(zzwx) SWZ4(zzwy) SWZ4(zzwz) SWZ4(zzww)

        SWZ4(zwxx) SWZ4(zwxy) SWZ4(zwxz) SWZ4(zwxw) SWZ4(zwyx) SWZ4(zwyy) SWZ4(zwyz) SWZ4(zwyw)
        SWZ4(zwzx) SWZ4(zwzy) SWZ4(zwzz) SWZ4(zwzw) SWZ4(zwwx) SWZ4(zwwy) SWZ4(zwwz) SWZ4(zwww)

        SWZ4(wxxx) SWZ4(wxxy) SWZ4(wxxz) SWZ4(wxxw) SWZ4(wxyx) SWZ4(wxyy) SWZ4(wxyz) SWZ4(wxyw)
        SWZ4(wxzx) SWZ4(wxzy) SWZ4(wxzz) SWZ4(wxzw) SWZ4(wxwx) SWZ4(wxwy) SWZ4(wxwz) SWZ4(wxww)

        SWZ4(wyxx) SWZ4(wyxy) SWZ4(wyxz) SWZ4(wyxw) SWZ4(wyyx) SWZ4(wyyy) SWZ4(wyyz) SWZ4(wyyw)
        SWZ4(wyzx) SWZ4(wyzy) SWZ4(wyzz) SWZ4(wyzw) SWZ4(wywx) SWZ4(wywy) SWZ4(wywz) SWZ4(wyww)

        SWZ4(wzxx) SWZ4(wzxy) SWZ4(wzxz) SWZ4(wzxw) SWZ4(wzyx) SWZ4(wzyy) SWZ4(wzyz) SWZ4(wzyw)
        SWZ4(wzzx) SWZ4(wzzy) SWZ4(wzzz) SWZ4(wzzw) SWZ4(wzwx) SWZ4(wzwy) SWZ4(wzwz) SWZ4(wzww)

        SWZ4(wwxx) SWZ4(wwxy) SWZ4(wwxz) SWZ4(wwxw) SWZ4(wwyx) SWZ4(wwyy) SWZ4(wwyz) SWZ4(wwyw)
        SWZ4(wwzx) SWZ4(wwzy) SWZ4(wwzz) SWZ4(wwzw) SWZ4(wwwx) SWZ4(wwwy) SWZ4(wwwz) SWZ4(wwww)
        /* clang-format on */

    private:
        // Helper to convert component to string
        template<typename T>
        static std::string ComponentToString(T&& val) {
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

#undef SWZ2
#undef SWZ3
#undef SWZ4
#undef MEM

    // ============================================================================
    // Var-Expr and Expr-Var Cross Operators for Vector Types
    // ============================================================================
    
    // Vec2: Var op Expr
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const VarBase<Math::Vec2> &lhs, const Expr<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const VarBase<Math::Vec2> &lhs, const Expr<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Vec2> &lhs, const Expr<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const VarBase<Math::Vec2> &lhs, const Expr<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec2: Expr op Var
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const Expr<Math::Vec2> &lhs, const VarBase<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const Expr<Math::Vec2> &lhs, const VarBase<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const Expr<Math::Vec2> &lhs, const VarBase<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const Expr<Math::Vec2> &lhs, const VarBase<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // Vec3: Var op Expr
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const VarBase<Math::Vec3> &lhs, const Expr<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const VarBase<Math::Vec3> &lhs, const Expr<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Vec3> &lhs, const Expr<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const VarBase<Math::Vec3> &lhs, const Expr<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec3: Expr op Var
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const Expr<Math::Vec3> &lhs, const VarBase<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const Expr<Math::Vec3> &lhs, const VarBase<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const Expr<Math::Vec3> &lhs, const VarBase<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const Expr<Math::Vec3> &lhs, const VarBase<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // Vec4: Var op Expr
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const VarBase<Math::Vec4> &lhs, const Expr<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const VarBase<Math::Vec4> &lhs, const Expr<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Vec4> &lhs, const Expr<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const VarBase<Math::Vec4> &lhs, const Expr<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec4: Expr op Var
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const Expr<Math::Vec4> &lhs, const VarBase<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const Expr<Math::Vec4> &lhs, const VarBase<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const Expr<Math::Vec4> &lhs, const VarBase<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const Expr<Math::Vec4> &lhs, const VarBase<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // ============================================================================
    // Vector * Scalar (Var<float>/Expr<float>) Mixed Operations
    // ============================================================================
    
    // Vec2 * Scalar
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const VarBase<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const VarBase<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const VarBase<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const VarBase<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const VarBase<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const VarBase<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Scalar * Vec2
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<float> &lhs, const VarBase<Math::Vec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const Expr<float> &lhs, const VarBase<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    // Vec2 * Scalar (Expr versions - to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const Expr<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const Expr<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const Expr<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const Expr<Math::Vec2> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), CloneNode(rhs)));
    }
    
    // Expr<Vec2> * VarBase<float> (to resolve ambiguity with VarBase<Vec2> * VarBase<float>)
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const Expr<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const Expr<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const Expr<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator/(const Expr<Math::Vec2> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // VarBase<float> * Expr<Vec2> (to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<float> &lhs, const Expr<Math::Vec2> &rhs) {
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec3 * Scalar
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const VarBase<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const VarBase<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const VarBase<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const VarBase<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const VarBase<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const VarBase<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Scalar * Vec3
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<float> &lhs, const VarBase<Math::Vec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const Expr<float> &lhs, const VarBase<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    // Vec3 * Scalar (Expr versions - to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const Expr<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const Expr<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const Expr<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const Expr<Math::Vec3> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), CloneNode(rhs)));
    }
    
    // Expr<Vec3> * VarBase<float> (to resolve ambiguity with VarBase<Vec3> * VarBase<float>)
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const Expr<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const Expr<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const Expr<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator/(const Expr<Math::Vec3> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // VarBase<float> * Expr<Vec3> (to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<float> &lhs, const Expr<Math::Vec3> &rhs) {
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec4 * Scalar
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const VarBase<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const VarBase<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const VarBase<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const VarBase<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const VarBase<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const VarBase<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    // Vec4 * Scalar (Expr versions - to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const Expr<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const Expr<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const Expr<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const Expr<Math::Vec4> &lhs, const Expr<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), CloneNode(rhs)));
    }

    // ============================================================================
    // Vector Compound Assignment with Expr
    // ============================================================================

    // Vec2 compound assignment with Expr<float> (scalar)
    [[nodiscard]] inline Var<Math::Vec2> &operator*=(Var<Math::Vec2> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::MulAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }
    [[nodiscard]] inline Var<Math::Vec2> &operator/=(Var<Math::Vec2> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::DivAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }

    // Vec3 compound assignment with Expr<float> (scalar)
    [[nodiscard]] inline Var<Math::Vec3> &operator*=(Var<Math::Vec3> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::MulAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }
    [[nodiscard]] inline Var<Math::Vec3> &operator/=(Var<Math::Vec3> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::DivAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }

    // Vec4 compound assignment with Expr<float> (scalar)
    [[nodiscard]] inline Var<Math::Vec4> &operator*=(Var<Math::Vec4> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::MulAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }
    [[nodiscard]] inline Var<Math::Vec4> &operator/=(Var<Math::Vec4> &lhs, const Expr<float> &rhs) {
        auto lhsLoad = lhs.Load();
        auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
            Node::CompoundAssignmentCode::DivAssign,
            std::move(lhsLoad),
            CloneNode(rhs)
        );
        Builder::Builder::Get().Build(*comAssign, true);
        return lhs;
    }

    // Expr<Vec4> * VarBase<float> (to resolve ambiguity with VarBase<Vec4> * VarBase<float>)
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const Expr<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const Expr<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const Expr<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator/(const Expr<Math::Vec4> &lhs, const VarBase<float> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    // VarBase<float> * Expr<Vec4> (to resolve ambiguity)
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<float> &lhs, const Expr<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // CPU Vec * GPU Scalar (for mixed operations like cameraRight * u)
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const Math::Vec2 &lhs, const VarBase<float> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsUniform), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const Math::Vec3 &lhs, const VarBase<float> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsUniform), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const Math::Vec4 &lhs, const VarBase<float> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsUniform), rhs.Load()));
    }

    // CPU Vec + GPU Vec (for mixed operations like cameraForward + cameraRight * u)
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const Math::Vec2 &lhs, const Expr<Math::Vec2> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsUniform), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator+(const Expr<Math::Vec2> &lhs, const Math::Vec2 &rhs) {
        auto rhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), std::move(rhsUniform)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const Math::Vec3 &lhs, const Expr<Math::Vec3> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsUniform), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator+(const Expr<Math::Vec3> &lhs, const Math::Vec3 &rhs) {
        auto rhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), std::move(rhsUniform)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const Math::Vec4 &lhs, const Expr<Math::Vec4> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsUniform), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator+(const Expr<Math::Vec4> &lhs, const Math::Vec4 &rhs) {
        auto rhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), std::move(rhsUniform)));
    }
    
    // CPU Vec - GPU Vec
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const Math::Vec2 &lhs, const Expr<Math::Vec2> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsUniform), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec2> operator-(const Expr<Math::Vec2> &lhs, const Math::Vec2 &rhs) {
        auto rhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Vec2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), std::move(rhsUniform)));
    }
    [[nodiscard]] inline Expr<Math::Vec3> operator-(const Math::Vec3 &lhs, const Expr<Math::Vec3> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsUniform), CloneNode(rhs)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator-(const Math::Vec4 &lhs, const Expr<Math::Vec4> &rhs) {
        auto lhsUniform = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsUniform), CloneNode(rhs)));
    }
    
    // Scalar * Vec4
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<float> &lhs, const VarBase<Math::Vec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const Expr<float> &lhs, const VarBase<Math::Vec4> &rhs) {
        return Expr<Math::Vec4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
}

#endif //EASYGPU_VARVECTOR_H
