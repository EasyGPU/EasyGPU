/**
 * VarMatrix.h:
 *      @Descripiton    :   The specified variable API for matrix
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_VARMATRIX_H
#define EASYGPU_VARMATRIX_H

// Note: This file should only be included by Var.h after Var main template is defined
// Do not include Var.h here to avoid circular inclusion
#include <IR/Value/Expr.h>
#include <IR/Value/ExprMatrix.h>

#include <Utility/Matrix.h>

#include <format>

namespace GPU::IR::Value {
    template<>
    class Var<Math::Mat2> : public VarBase<Math::Mat2> {
    public:
        using VarBase<Math::Mat2>::VarBase;
        using VarBase<Math::Mat2>::Load;
        using VarBase<Math::Mat2>::operator=;

        // Column constructor: Var<Mat2> m(col0, col1)
        template<typename C0, typename C1>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec2>>)
        Var(C0&& col0, C1&& col1) : VarBase() {
            std::string c0Str = ColumnToString(std::forward<C0>(col0));
            std::string c1Str = ColumnToString(std::forward<C1>(col1));
            auto initCode = std::format("{}=mat2({}, {});\n", _varNode->VarName(), c0Str, c1Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec2> operator[](T Index) {
            return Var<Math::Vec2>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec2> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec2> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        // Helper to convert column to string
        template<typename T>
        static std::string ColumnToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat3> : public VarBase<Math::Mat3> {
    public:
        using VarBase<Math::Mat3>::VarBase;
        using VarBase<Math::Mat3>::Load;
        using VarBase<Math::Mat3>::operator=;

        // Column constructor: Var<Mat3> m(col0, col1, col2)
        template<typename C0, typename C1, typename C2>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec3>>)
        Var(C0&& col0, C1&& col1, C2&& col2) : VarBase() {
            std::string c0Str = ColumnToString(std::forward<C0>(col0));
            std::string c1Str = ColumnToString(std::forward<C1>(col1));
            std::string c2Str = ColumnToString(std::forward<C2>(col2));
            auto initCode = std::format("{}=mat3({}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec3> operator[](T Index) {
            return Var<Math::Vec3>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec3> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec3> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string ColumnToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat4> : public VarBase<Math::Mat4> {
    public:
        using VarBase<Math::Mat4>::VarBase;
        using VarBase<Math::Mat4>::Load;
        using VarBase<Math::Mat4>::operator=;

        // Column constructor: Var<Mat4> m(col0, col1, col2, col3)
        template<typename C0, typename C1, typename C2, typename C3>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C3>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C3>, Expr<Math::Vec4>>)
        Var(C0&& col0, C1&& col1, C2&& col2, C3&& col3) : VarBase() {
            std::string c0Str = ColumnToString(std::forward<C0>(col0));
            std::string c1Str = ColumnToString(std::forward<C1>(col1));
            std::string c2Str = ColumnToString(std::forward<C2>(col2));
            std::string c3Str = ColumnToString(std::forward<C3>(col3));
            auto initCode = std::format("{}=mat4({}, {}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str, c3Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec4> operator[](T Index) {
            return Var<Math::Vec4>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec4> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec4> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string ColumnToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    // ==================== Rectangular Matrices ====================
    
    template<>
    class Var<Math::Mat2x3> : public VarBase<Math::Mat2x3> {
    public:
        using VarBase<Math::Mat2x3>::VarBase;
        using VarBase<Math::Mat2x3>::Load;
        using VarBase<Math::Mat2x3>::operator=;

        // Column constructor: Var<Mat2x3> m(col0, col1) - 2 columns of vec3
        template<typename C0, typename C1>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec3>>)
        Var(C0&& col0, C1&& col1) : VarBase() {
            std::string c0Str = Vec3ToString(std::forward<C0>(col0));
            std::string c1Str = Vec3ToString(std::forward<C1>(col1));
            auto initCode = std::format("{}=mat2x3({}, {});\n", _varNode->VarName(), c0Str, c1Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec3> operator[](T Index) {
            return Var<Math::Vec3>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec3> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec3> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec3ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat2x4> : public VarBase<Math::Mat2x4> {
    public:
        using VarBase<Math::Mat2x4>::VarBase;
        using VarBase<Math::Mat2x4>::Load;
        using VarBase<Math::Mat2x4>::operator=;

        // Column constructor: Var<Mat2x4> m(col0, col1) - 2 columns of vec4
        template<typename C0, typename C1>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec4>>)
        Var(C0&& col0, C1&& col1) : VarBase() {
            std::string c0Str = Vec4ToString(std::forward<C0>(col0));
            std::string c1Str = Vec4ToString(std::forward<C1>(col1));
            auto initCode = std::format("{}=mat2x4({}, {});\n", _varNode->VarName(), c0Str, c1Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec4> operator[](T Index) {
            return Var<Math::Vec4>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec4> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec4> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec4ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat3x2> : public VarBase<Math::Mat3x2> {
    public:
        using VarBase<Math::Mat3x2>::VarBase;
        using VarBase<Math::Mat3x2>::Load;
        using VarBase<Math::Mat3x2>::operator=;

        // Column constructor: Var<Mat3x2> m(col0, col1, col2) - 3 columns of vec2
        template<typename C0, typename C1, typename C2>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec2>>)
        Var(C0&& col0, C1&& col1, C2&& col2) : VarBase() {
            std::string c0Str = Vec2ToString(std::forward<C0>(col0));
            std::string c1Str = Vec2ToString(std::forward<C1>(col1));
            std::string c2Str = Vec2ToString(std::forward<C2>(col2));
            auto initCode = std::format("{}=mat3x2({}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec2> operator[](T Index) {
            return Var<Math::Vec2>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec2> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec2> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec2ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat3x4> : public VarBase<Math::Mat3x4> {
    public:
        using VarBase<Math::Mat3x4>::VarBase;
        using VarBase<Math::Mat3x4>::Load;
        using VarBase<Math::Mat3x4>::operator=;

        // Column constructor: Var<Mat3x4> m(col0, col1, col2) - 3 columns of vec4
        template<typename C0, typename C1, typename C2>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec4>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec4>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec4>>)
        Var(C0&& col0, C1&& col1, C2&& col2) : VarBase() {
            std::string c0Str = Vec4ToString(std::forward<C0>(col0));
            std::string c1Str = Vec4ToString(std::forward<C1>(col1));
            std::string c2Str = Vec4ToString(std::forward<C2>(col2));
            auto initCode = std::format("{}=mat3x4({}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec4> operator[](T Index) {
            return Var<Math::Vec4>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec4> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec4> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec4ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec4>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat4x2> : public VarBase<Math::Mat4x2> {
    public:
        using VarBase<Math::Mat4x2>::VarBase;
        using VarBase<Math::Mat4x2>::Load;
        using VarBase<Math::Mat4x2>::operator=;

        // Column constructor: Var<Mat4x2> m(col0, col1, col2, col3) - 4 columns of vec2
        template<typename C0, typename C1, typename C2, typename C3>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec2>>) &&
                     (std::same_as<std::remove_cvref_t<C3>, Var<Math::Vec2>> || std::same_as<std::remove_cvref_t<C3>, Expr<Math::Vec2>>)
        Var(C0&& col0, C1&& col1, C2&& col2, C3&& col3) : VarBase() {
            std::string c0Str = Vec2ToString(std::forward<C0>(col0));
            std::string c1Str = Vec2ToString(std::forward<C1>(col1));
            std::string c2Str = Vec2ToString(std::forward<C2>(col2));
            std::string c3Str = Vec2ToString(std::forward<C3>(col3));
            auto initCode = std::format("{}=mat4x2({}, {}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str, c3Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec2> operator[](T Index) {
            return Var<Math::Vec2>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec2> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec2> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec2ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec2>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    template<>
    class Var<Math::Mat4x3> : public VarBase<Math::Mat4x3> {
    public:
        using VarBase<Math::Mat4x3>::VarBase;
        using VarBase<Math::Mat4x3>::Load;
        using VarBase<Math::Mat4x3>::operator=;

        // Column constructor: Var<Mat4x3> m(col0, col1, col2, col3) - 4 columns of vec3
        template<typename C0, typename C1, typename C2, typename C3>
            requires (std::same_as<std::remove_cvref_t<C0>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C0>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C1>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C1>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C2>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C2>, Expr<Math::Vec3>>) &&
                     (std::same_as<std::remove_cvref_t<C3>, Var<Math::Vec3>> || std::same_as<std::remove_cvref_t<C3>, Expr<Math::Vec3>>)
        Var(C0&& col0, C1&& col1, C2&& col2, C3&& col3) : VarBase() {
            std::string c0Str = Vec3ToString(std::forward<C0>(col0));
            std::string c1Str = Vec3ToString(std::forward<C1>(col1));
            std::string c2Str = Vec3ToString(std::forward<C2>(col2));
            std::string c3Str = Vec3ToString(std::forward<C3>(col3));
            auto initCode = std::format("{}=mat4x3({}, {}, {}, {});\n", _varNode->VarName(), c0Str, c1Str, c2Str, c3Str);
            Builder::Builder::Get().Context()->PushTranslatedCode(initCode);
        }

    public:
        template<CountableType T>
        Var<Math::Vec3> operator[](T Index) {
            return Var<Math::Vec3>(std::format("{}[{}]", _varNode->VarName(), ValueToString(Index)));
        }
        
        Var<Math::Vec3> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

        template<ScalarType IndexT>
        Var<Math::Vec3> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return {std::format("{}[{}]", _varNode->VarName(), exprStr)};
        }

    private:
        template<typename T>
        static std::string Vec3ToString(T&& val) {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::same_as<U, Expr<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Node());
            } else if constexpr (std::same_as<U, Var<Math::Vec3>>) {
                return Builder::Builder::Get().BuildNode(*val.Load().get());
            } else {
                return "";
            }
        }
    };

    // ==================== Var Matrix-Vector Multiplication ====================
    // Mat2 * Vec2
    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Mat2> &lhs, const VarBase<Math::Vec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Mat3 * Vec3
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Mat3> &lhs, const VarBase<Math::Vec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Mat4 * Vec4
    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Mat4> &lhs, const VarBase<Math::Vec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Rectangular matrix * vector
    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Mat2x3> &lhs, const VarBase<Math::Vec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Mat3x2> &lhs, const VarBase<Math::Vec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Mat2x4> &lhs, const VarBase<Math::Vec2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Vec2> operator*(const VarBase<Math::Mat4x2> &lhs, const VarBase<Math::Vec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Vec4> operator*(const VarBase<Math::Mat3x4> &lhs, const VarBase<Math::Vec3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Vec3> operator*(const VarBase<Math::Mat4x3> &lhs, const VarBase<Math::Vec4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Vec3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // ============================================================================
    // Var<int> * Matrix Mixed Operations (int变量与float矩阵)
    // ============================================================================
    
    // Var<int> * Mat2
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const VarBase<int> &lhs, const Expr<Math::Mat2> &rhs) {
        return Expr<Math::Mat2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // Mat2 * Var<int>
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const VarBase<Math::Mat2> &lhs, const VarBase<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const Expr<Math::Mat2> &lhs, const VarBase<int> &rhs) {
        return Expr<Math::Mat2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    // Var<int> * Mat3
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const VarBase<int> &lhs, const Expr<Math::Mat3> &rhs) {
        return Expr<Math::Mat3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // Mat3 * Var<int>
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const VarBase<Math::Mat3> &lhs, const VarBase<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const Expr<Math::Mat3> &lhs, const VarBase<int> &rhs) {
        return Expr<Math::Mat3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    // Var<int> * Mat4
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const VarBase<int> &lhs, const Expr<Math::Mat4> &rhs) {
        return Expr<Math::Mat4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    // Mat4 * Var<int>
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const VarBase<Math::Mat4> &lhs, const VarBase<int> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const Expr<Math::Mat4> &lhs, const VarBase<int> &rhs) {
        return Expr<Math::Mat4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    // Rectangular matrices with Var<int>
    [[nodiscard]] inline Expr<Math::Mat2x3> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat2x3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2x3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2x4> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat2x4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2x4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3x2> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat3x2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3x2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3x4> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat3x4> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3x4>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4x2> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat4x2> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4x2>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4x3> operator*(const VarBase<int> &lhs, const VarBase<Math::Mat4x3> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4x3>(std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
}

#endif //EASYGPU_VARMATRIX_H
