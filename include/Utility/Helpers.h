#pragma once

/**
 * Helpers.h:
 *      @Descripiton    :   Helper functions and type aliases for EasyGPU
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_HELPERS_H
#define EASYGPU_HELPERS_H

#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Value/ExprVector.h>
#include <IR/Value/ExprIVector.h>
#include <IR/Value/ExprMatrix.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Node/CallInst.h>
#include <IR/Builder/Builder.h>
#include <Utility/Math.h>

#include <format>

namespace GPU {
    // ============================================================================
    // Type Aliases for Var types
    // ============================================================================
    namespace Alias {
        // Scalar types
        using Int   = IR::Value::Var<int>;
        using Float = IR::Value::Var<float>;
        using Bool  = IR::Value::Var<bool>;

        // Float vector types
        using Float2 = IR::Value::Var<Math::Vec2>;
        using Float3 = IR::Value::Var<Math::Vec3>;
        using Float4 = IR::Value::Var<Math::Vec4>;

        // Int vector types
        using Int2 = IR::Value::Var<Math::IVec2>;
        using Int3 = IR::Value::Var<Math::IVec3>;
        using Int4 = IR::Value::Var<Math::IVec4>;

        // Matrix types
        using Matrix2 = IR::Value::Var<Math::Mat2>;
        using Matrix3 = IR::Value::Var<Math::Mat3>;
        using Matrix4 = IR::Value::Var<Math::Mat4>;
        using Matrix2x3 = IR::Value::Var<Math::Mat2x3>;
        using Matrix3x2 = IR::Value::Var<Math::Mat3x2>;
        using Matrix2x4 = IR::Value::Var<Math::Mat2x4>;
        using Matrix4x2 = IR::Value::Var<Math::Mat4x2>;
        using Matrix3x4 = IR::Value::Var<Math::Mat3x4>;
        using Matrix4x3 = IR::Value::Var<Math::Mat4x3>;

        // Expr type aliases
        using IntExpr   = IR::Value::Expr<int>;
        using FloatExpr = IR::Value::Expr<float>;
        using BoolExpr  = IR::Value::Expr<bool>;
        using Float2Expr = IR::Value::Expr<Math::Vec2>;
        using Float3Expr = IR::Value::Expr<Math::Vec3>;
        using Float4Expr = IR::Value::Expr<Math::Vec4>;
        using Int2Expr = IR::Value::Expr<Math::IVec2>;
        using Int3Expr = IR::Value::Expr<Math::IVec3>;
        using Int4Expr = IR::Value::Expr<Math::IVec4>;

        // Matrix Expr type aliases
        using Mat2Expr = IR::Value::Expr<Math::Mat2>;
        using Mat3Expr = IR::Value::Expr<Math::Mat3>;
        using Mat4Expr = IR::Value::Expr<Math::Mat4>;
        using Mat2x3Expr = IR::Value::Expr<Math::Mat2x3>;
        using Mat3x2Expr = IR::Value::Expr<Math::Mat3x2>;
        using Mat2x4Expr = IR::Value::Expr<Math::Mat2x4>;
        using Mat4x2Expr = IR::Value::Expr<Math::Mat4x2>;
        using Mat3x4Expr = IR::Value::Expr<Math::Mat3x4>;
        using Mat4x3Expr = IR::Value::Expr<Math::Mat4x3>;
    }

    // ============================================================================
    // Helper functions to construct vectors from components
    // Uses perfect forwarding to accept Var, Expr, or literal values
    // ============================================================================
    namespace Construct {
        // ============================================================================
        // Type traits and concepts for Make function constraints
        // ============================================================================
        namespace Detail {
            // Helper to build parameter list for vector constructors
            inline std::vector<std::unique_ptr<IR::Node::Node>> BuildVectorParams(
                const std::vector<IR::Value::ExprBase*>& exprs) {
                std::vector<std::unique_ptr<IR::Node::Node>> params;
                params.reserve(exprs.size());
                for (auto* expr : exprs) {
                    params.push_back(const_cast<IR::Value::ExprBase*>(expr)->Node()->Clone());
                }
                return params;
            }

            // Trait to detect if T is Var<U>
            template<typename T, typename U>
            struct IsVarOf : std::false_type {};
            template<typename U>
            struct IsVarOf<IR::Value::Var<U>, U> : std::true_type {};

            // Trait to detect if T is Expr<U>
            template<typename T, typename U>
            struct IsExprOf : std::false_type {};
            template<typename U>
            struct IsExprOf<IR::Value::Expr<U>, U> : std::true_type {};

            // Concept for valid float component type (Var<float>, Expr<float>, or convertible-to-float literal)
            template<typename T>
            concept FloatComponent = 
                std::convertible_to<std::remove_cvref_t<T>, float> ||
                IsVarOf<std::remove_cvref_t<T>, float>::value ||
                IsExprOf<std::remove_cvref_t<T>, float>::value;

            // Concept for valid int component type (Var<int>, Expr<int>, or convertible-to-int literal)
            template<typename T>
            concept IntComponent = 
                std::convertible_to<std::remove_cvref_t<T>, int> ||
                IsVarOf<std::remove_cvref_t<T>, int>::value ||
                IsExprOf<std::remove_cvref_t<T>, int>::value;

            // Concept for valid Vec2 column type (Var<Vec2>, Expr<Vec2>, or Math::Vec2)
            template<typename T>
            concept Vec2Component = 
                std::same_as<std::remove_cvref_t<T>, Math::Vec2> ||
                IsVarOf<std::remove_cvref_t<T>, Math::Vec2>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::Vec2>::value;

            // Concept for valid Vec3 column type (Var<Vec3>, Expr<Vec3>, or Math::Vec3)
            template<typename T>
            concept Vec3Component = 
                std::same_as<std::remove_cvref_t<T>, Math::Vec3> ||
                IsVarOf<std::remove_cvref_t<T>, Math::Vec3>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::Vec3>::value;

            // Concept for valid Vec4 column type (Var<Vec4>, Expr<Vec4>, or Math::Vec4)
            template<typename T>
            concept Vec4Component = 
                std::same_as<std::remove_cvref_t<T>, Math::Vec4> ||
                IsVarOf<std::remove_cvref_t<T>, Math::Vec4>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::Vec4>::value;

            // Concept for valid IVec2 component type (Var<IVec2>, Expr<IVec2>, or Math::IVec2)
            template<typename T>
            concept IVec2Component = 
                std::same_as<std::remove_cvref_t<T>, Math::IVec2> ||
                IsVarOf<std::remove_cvref_t<T>, Math::IVec2>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::IVec2>::value;

            // Concept for valid IVec3 component type (Var<IVec3>, Expr<IVec3>, or Math::IVec3)
            template<typename T>
            concept IVec3Component = 
                std::same_as<std::remove_cvref_t<T>, Math::IVec3> ||
                IsVarOf<std::remove_cvref_t<T>, Math::IVec3>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::IVec3>::value;

            // Concept for valid IVec4 component type (Var<IVec4>, Expr<IVec4>, or Math::IVec4)
            template<typename T>
            concept IVec4Component = 
                std::same_as<std::remove_cvref_t<T>, Math::IVec4> ||
                IsVarOf<std::remove_cvref_t<T>, Math::IVec4>::value ||
                IsExprOf<std::remove_cvref_t<T>, Math::IVec4>::value;

            // Internal implementations that take Expr
            [[nodiscard]] inline IR::Value::Expr<Math::Vec2> MakeFloat2Impl(
                const IR::Value::Expr<float>& x,
                const IR::Value::Expr<float>& y) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<float>*>(&x),
                    const_cast<IR::Value::Expr<float>*>(&y)
                };
                return Math::MakeCall<Math::Vec2>("vec2", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Vec3> MakeFloat3Impl(
                const IR::Value::Expr<float>& x,
                const IR::Value::Expr<float>& y,
                const IR::Value::Expr<float>& z) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<float>*>(&x),
                    const_cast<IR::Value::Expr<float>*>(&y),
                    const_cast<IR::Value::Expr<float>*>(&z)
                };
                return Math::MakeCall<Math::Vec3>("vec3", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4Impl(
                const IR::Value::Expr<float>& x,
                const IR::Value::Expr<float>& y,
                const IR::Value::Expr<float>& z,
                const IR::Value::Expr<float>& w) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<float>*>(&x),
                    const_cast<IR::Value::Expr<float>*>(&y),
                    const_cast<IR::Value::Expr<float>*>(&z),
                    const_cast<IR::Value::Expr<float>*>(&w)
                };
                return Math::MakeCall<Math::Vec4>("vec4", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::IVec2> MakeInt2Impl(
                const IR::Value::Expr<int>& x,
                const IR::Value::Expr<int>& y) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<int>*>(&x),
                    const_cast<IR::Value::Expr<int>*>(&y)
                };
                return Math::MakeCall<Math::IVec2>("ivec2", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::IVec3> MakeInt3Impl(
                const IR::Value::Expr<int>& x,
                const IR::Value::Expr<int>& y,
                const IR::Value::Expr<int>& z) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<int>*>(&x),
                    const_cast<IR::Value::Expr<int>*>(&y),
                    const_cast<IR::Value::Expr<int>*>(&z)
                };
                return Math::MakeCall<Math::IVec3>("ivec3", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4Impl(
                const IR::Value::Expr<int>& x,
                const IR::Value::Expr<int>& y,
                const IR::Value::Expr<int>& z,
                const IR::Value::Expr<int>& w) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<int>*>(&x),
                    const_cast<IR::Value::Expr<int>*>(&y),
                    const_cast<IR::Value::Expr<int>*>(&z),
                    const_cast<IR::Value::Expr<int>*>(&w)
                };
                return Math::MakeCall<Math::IVec4>("ivec4", BuildVectorParams(exprs));
            }

            // ============================================================================
            // Broadcast construction internal implementations
            // ============================================================================

            // MakeFloat3 from Vec2 + scalar - direct implementation using GLSL constructor syntax
            [[nodiscard]] inline IR::Value::Expr<Math::Vec3> MakeFloat3BroadcastImpl(
                const IR::Value::ExprBase& xy,
                const IR::Value::ExprBase& z) {
                std::string xyStr = IR::Builder::Builder::Get().BuildNode(*xy.Node());
                std::string zStr = IR::Builder::Builder::Get().BuildNode(*z.Node());
                return IR::Value::Expr<Math::Vec3>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("vec3(({}).xy, ({}).x)", xyStr, zStr)));
            }

            // MakeFloat4 from Vec2 + scalar + scalar
            [[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4BroadcastImpl(
                const IR::Value::ExprBase& xy,
                const IR::Value::ExprBase& z,
                const IR::Value::ExprBase& w) {
                std::string xyStr = IR::Builder::Builder::Get().BuildNode(*xy.Node());
                std::string zStr = IR::Builder::Builder::Get().BuildNode(*z.Node());
                std::string wStr = IR::Builder::Builder::Get().BuildNode(*w.Node());
                return IR::Value::Expr<Math::Vec4>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("vec4(({}).xy, ({}).x, ({}).x)", xyStr, zStr, wStr)));
            }

            // MakeFloat4 from Vec3 + scalar
            [[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4BroadcastImpl(
                const IR::Value::ExprBase& xyz,
                const IR::Value::ExprBase& w) {
                std::string xyzStr = IR::Builder::Builder::Get().BuildNode(*xyz.Node());
                std::string wStr = IR::Builder::Builder::Get().BuildNode(*w.Node());
                return IR::Value::Expr<Math::Vec4>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("vec4(({}).xyz, ({}).x)", xyzStr, wStr)));
            }

            // MakeInt3 from IVec2 + scalar
            [[nodiscard]] inline IR::Value::Expr<Math::IVec3> MakeInt3BroadcastImpl(
                const IR::Value::ExprBase& xy,
                const IR::Value::ExprBase& z) {
                std::string xyStr = IR::Builder::Builder::Get().BuildNode(*xy.Node());
                std::string zStr = IR::Builder::Builder::Get().BuildNode(*z.Node());
                return IR::Value::Expr<Math::IVec3>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("ivec3(({}).xy, ({}).x)", xyStr, zStr)));
            }

            // MakeInt4 from IVec2 + scalar + scalar
            [[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4BroadcastImpl(
                const IR::Value::ExprBase& xy,
                const IR::Value::ExprBase& z,
                const IR::Value::ExprBase& w) {
                std::string xyStr = IR::Builder::Builder::Get().BuildNode(*xy.Node());
                std::string zStr = IR::Builder::Builder::Get().BuildNode(*z.Node());
                std::string wStr = IR::Builder::Builder::Get().BuildNode(*w.Node());
                return IR::Value::Expr<Math::IVec4>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("ivec4(({}).xy, ({}).x, ({}).x)", xyStr, zStr, wStr)));
            }

            // MakeInt4 from IVec3 + scalar
            [[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4BroadcastImpl(
                const IR::Value::ExprBase& xyz,
                const IR::Value::ExprBase& w) {
                std::string xyzStr = IR::Builder::Builder::Get().BuildNode(*xyz.Node());
                std::string wStr = IR::Builder::Builder::Get().BuildNode(*w.Node());
                return IR::Value::Expr<Math::IVec4>(
                    std::make_unique<IR::Node::LoadUniformNode>(
                        std::format("ivec4(({}).xyz, ({}).x)", xyzStr, wStr)));
            }
        }

        // Public template versions with perfect forwarding and type constraints
        // Multi-component versions
        template<typename X, typename Y>
            requires Detail::FloatComponent<X> && Detail::FloatComponent<Y>
        [[nodiscard]] inline auto MakeFloat2(X&& x, Y&& y) {
            return Detail::MakeFloat2Impl(
                IR::Value::Expr<float>(std::forward<X>(x)),
                IR::Value::Expr<float>(std::forward<Y>(y)));
        }

        template<typename X, typename Y, typename Z>
            requires Detail::FloatComponent<X> && Detail::FloatComponent<Y> && Detail::FloatComponent<Z>
        [[nodiscard]] inline auto MakeFloat3(X&& x, Y&& y, Z&& z) {
            return Detail::MakeFloat3Impl(
                IR::Value::Expr<float>(std::forward<X>(x)),
                IR::Value::Expr<float>(std::forward<Y>(y)),
                IR::Value::Expr<float>(std::forward<Z>(z)));
        }

        template<typename X, typename Y, typename Z, typename W>
            requires Detail::FloatComponent<X> && Detail::FloatComponent<Y> && 
                     Detail::FloatComponent<Z> && Detail::FloatComponent<W>
        [[nodiscard]] inline auto MakeFloat4(X&& x, Y&& y, Z&& z, W&& w) {
            return Detail::MakeFloat4Impl(
                IR::Value::Expr<float>(std::forward<X>(x)),
                IR::Value::Expr<float>(std::forward<Y>(y)),
                IR::Value::Expr<float>(std::forward<Z>(z)),
                IR::Value::Expr<float>(std::forward<W>(w)));
        }

        template<typename X, typename Y>
            requires Detail::IntComponent<X> && Detail::IntComponent<Y>
        [[nodiscard]] inline auto MakeInt2(X&& x, Y&& y) {
            return Detail::MakeInt2Impl(
                IR::Value::Expr<int>(std::forward<X>(x)),
                IR::Value::Expr<int>(std::forward<Y>(y)));
        }

        template<typename X, typename Y, typename Z>
            requires Detail::IntComponent<X> && Detail::IntComponent<Y> && Detail::IntComponent<Z>
        [[nodiscard]] inline auto MakeInt3(X&& x, Y&& y, Z&& z) {
            return Detail::MakeInt3Impl(
                IR::Value::Expr<int>(std::forward<X>(x)),
                IR::Value::Expr<int>(std::forward<Y>(y)),
                IR::Value::Expr<int>(std::forward<Z>(z)));
        }

        template<typename X, typename Y, typename Z, typename W>
            requires Detail::IntComponent<X> && Detail::IntComponent<Y> && 
                     Detail::IntComponent<Z> && Detail::IntComponent<W>
        [[nodiscard]] inline auto MakeInt4(X&& x, Y&& y, Z&& z, W&& w) {
            return Detail::MakeInt4Impl(
                IR::Value::Expr<int>(std::forward<X>(x)),
                IR::Value::Expr<int>(std::forward<Y>(y)),
                IR::Value::Expr<int>(std::forward<Z>(z)),
                IR::Value::Expr<int>(std::forward<W>(w)));
        }

        // ============================================================================
        // Broadcast construction from lower-dimensional vectors
        // ============================================================================

        // Local trait to detect Expr types in this namespace
        template<typename T>
        struct IsExprT : std::false_type {};
        template<typename U>
        struct IsExprT<IR::Value::Expr<U>> : std::true_type {};
        template<typename T>
        inline constexpr bool IsExpr_v = IsExprT<T>::value;

        // MakeFloat3 from Float2/Float2Expr + float
        template<typename XY, typename Z>
            requires Detail::Vec2Component<XY> && 
                     (std::convertible_to<std::remove_cvref_t<Z>, float> || 
                      Detail::IsVarOf<std::remove_cvref_t<Z>, float>::value ||
                      Detail::IsExprOf<std::remove_cvref_t<Z>, float>::value)
        [[nodiscard]] inline auto MakeFloat3(XY&& xy, Z&& z) {
            IR::Value::Expr<float> zExpr = Detail::ToExpr(std::forward<Z>(z));
            if constexpr (IsExpr_v<std::remove_cvref_t<XY>>) {
                return Detail::MakeFloat3BroadcastImpl(xy, std::move(zExpr));
            } else {
                return Detail::MakeFloat3BroadcastImpl(
                    IR::Value::Expr<Math::Vec2>(std::forward<XY>(xy)),
                    std::move(zExpr));
            }
        }

        // MakeFloat4 from Float2/Float2Expr + float + float
        template<typename XY, typename Z, typename W>
            requires Detail::Vec2Component<XY> && 
                     (std::convertible_to<std::remove_cvref_t<Z>, float> || Detail::IsVarOf<std::remove_cvref_t<Z>, float>::value || Detail::IsExprOf<std::remove_cvref_t<Z>, float>::value) &&
                     (std::convertible_to<std::remove_cvref_t<W>, float> || Detail::IsVarOf<std::remove_cvref_t<W>, float>::value || Detail::IsExprOf<std::remove_cvref_t<W>, float>::value)
        [[nodiscard]] inline auto MakeFloat4(XY&& xy, Z&& z, W&& w) {
            IR::Value::Expr<float> zExpr = Detail::ToExpr(std::forward<Z>(z));
            IR::Value::Expr<float> wExpr = Detail::ToExpr(std::forward<W>(w));
            if constexpr (IsExpr_v<std::remove_cvref_t<XY>>) {
                return Detail::MakeFloat4BroadcastImpl(xy, std::move(zExpr), std::move(wExpr));
            } else {
                return Detail::MakeFloat4BroadcastImpl(
                    IR::Value::Expr<Math::Vec2>(std::forward<XY>(xy)),
                    std::move(zExpr), std::move(wExpr));
            }
        }

        // MakeFloat4 from Float3/Float3Expr + float
        template<typename XYZ, typename W>
            requires Detail::Vec3Component<XYZ> && 
                     (std::convertible_to<std::remove_cvref_t<W>, float> || Detail::IsVarOf<std::remove_cvref_t<W>, float>::value || Detail::IsExprOf<std::remove_cvref_t<W>, float>::value)
        [[nodiscard]] inline auto MakeFloat4(XYZ&& xyz, W&& w) {
            IR::Value::Expr<float> wExpr = Detail::ToExpr(std::forward<W>(w));
            if constexpr (IsExpr_v<std::remove_cvref_t<XYZ>>) {
                return Detail::MakeFloat4BroadcastImpl(xyz, std::move(wExpr));
            } else {
                return Detail::MakeFloat4BroadcastImpl(
                    IR::Value::Expr<Math::Vec3>(std::forward<XYZ>(xyz)),
                    std::move(wExpr));
            }
        }

        // MakeInt3 from Int2/Int2Expr + int
        template<typename XY, typename Z>
            requires Detail::IVec2Component<XY> && 
                     (std::convertible_to<std::remove_cvref_t<Z>, int> || 
                      Detail::IsVarOf<std::remove_cvref_t<Z>, int>::value ||
                      Detail::IsExprOf<std::remove_cvref_t<Z>, int>::value)
        [[nodiscard]] inline auto MakeInt3(XY&& xy, Z&& z) {
            IR::Value::Expr<int> zExpr = Detail::ToExpr(std::forward<Z>(z));
            if constexpr (IsExpr_v<std::remove_cvref_t<XY>>) {
                return Detail::MakeInt3BroadcastImpl(xy, std::move(zExpr));
            } else {
                return Detail::MakeInt3BroadcastImpl(
                    IR::Value::Expr<Math::IVec2>(std::forward<XY>(xy)),
                    std::move(zExpr));
            }
        }

        // MakeInt4 from Int2/Int2Expr + int + int
        template<typename XY, typename Z, typename W>
            requires Detail::IVec2Component<XY> && 
                     (std::convertible_to<std::remove_cvref_t<Z>, int> || Detail::IsVarOf<std::remove_cvref_t<Z>, int>::value || Detail::IsExprOf<std::remove_cvref_t<Z>, int>::value) &&
                     (std::convertible_to<std::remove_cvref_t<W>, int> || Detail::IsVarOf<std::remove_cvref_t<W>, int>::value || Detail::IsExprOf<std::remove_cvref_t<W>, int>::value)
        [[nodiscard]] inline auto MakeInt4(XY&& xy, Z&& z, W&& w) {
            IR::Value::Expr<int> zExpr = Detail::ToExpr(std::forward<Z>(z));
            IR::Value::Expr<int> wExpr = Detail::ToExpr(std::forward<W>(w));
            if constexpr (IsExpr_v<std::remove_cvref_t<XY>>) {
                return Detail::MakeInt4BroadcastImpl(xy, std::move(zExpr), std::move(wExpr));
            } else {
                return Detail::MakeInt4BroadcastImpl(
                    IR::Value::Expr<Math::IVec2>(std::forward<XY>(xy)),
                    std::move(zExpr), std::move(wExpr));
            }
        }

        // MakeInt4 from Int3/Int3Expr + int
        template<typename XYZ, typename W>
            requires Detail::IVec3Component<XYZ> && 
                     (std::convertible_to<std::remove_cvref_t<W>, int> || Detail::IsVarOf<std::remove_cvref_t<W>, int>::value || Detail::IsExprOf<std::remove_cvref_t<W>, int>::value)
        [[nodiscard]] inline auto MakeInt4(XYZ&& xyz, W&& w) {
            IR::Value::Expr<int> wExpr = Detail::ToExpr(std::forward<W>(w));
            if constexpr (IsExpr_v<std::remove_cvref_t<XYZ>>) {
                return Detail::MakeInt4BroadcastImpl(xyz, std::move(wExpr));
            } else {
                return Detail::MakeInt4BroadcastImpl(
                    IR::Value::Expr<Math::IVec3>(std::forward<XYZ>(xyz)),
                    std::move(wExpr));
            }
        }

        // Type trait to detect Vec types (for disabling lazy fill when Vec is passed)
        template<typename T>
        struct IsVecType : std::false_type {};
        template<> struct IsVecType<Math::Vec2> : std::true_type {};
        template<> struct IsVecType<Math::Vec3> : std::true_type {};
        template<> struct IsVecType<Math::Vec4> : std::true_type {};
        template<> struct IsVecType<Math::IVec2> : std::true_type {};
        template<> struct IsVecType<Math::IVec3> : std::true_type {};
        template<> struct IsVecType<Math::IVec4> : std::true_type {};

        // Single-component lazy fill versions (e.g., MakeFloat3(1.0) creates vec3(1.0, 1.0, 1.0))
        // Disabled when X is a Vec type to avoid ambiguity with the overloads below
        template<typename X>
            requires Detail::FloatComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeFloat2(X&& x) {
            IR::Value::Expr<float> val(std::forward<X>(x));
            return Detail::MakeFloat2Impl(val, val);
        }

        template<typename X>
            requires Detail::FloatComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeFloat3(X&& x) {
            IR::Value::Expr<float> val(std::forward<X>(x));
            return Detail::MakeFloat3Impl(val, val, val);
        }

        template<typename X>
            requires Detail::FloatComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeFloat4(X&& x) {
            IR::Value::Expr<float> val(std::forward<X>(x));
            return Detail::MakeFloat4Impl(val, val, val, val);
        }

        template<typename X>
            requires Detail::IntComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeInt2(X&& x) {
            IR::Value::Expr<int> val(std::forward<X>(x));
            return Detail::MakeInt2Impl(val, val);
        }

        template<typename X>
            requires Detail::IntComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeInt3(X&& x) {
            IR::Value::Expr<int> val(std::forward<X>(x));
            return Detail::MakeInt3Impl(val, val, val);
        }

        template<typename X>
            requires Detail::IntComponent<X> && (!IsVecType<std::remove_cvref_t<X>>::value)
        [[nodiscard]] inline auto MakeInt4(X&& x) {
            IR::Value::Expr<int> val(std::forward<X>(x));
            return Detail::MakeInt4Impl(val, val, val, val);
        }

        // CPU-side vector to GPU Expr vector conversions
        [[nodiscard]] inline IR::Value::Expr<Math::Vec2> MakeFloat2(const Math::Vec2& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::Vec2>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Vec3> MakeFloat3(const Math::Vec3& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::Vec3>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4(const Math::Vec4& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::Vec4>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::IVec2> MakeInt2(const Math::IVec2& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::IVec2>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::IVec3> MakeInt3(const Math::IVec3& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::IVec3>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4(const Math::IVec4& v) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(v));
            return IR::Value::Expr<Math::IVec4>(std::move(uniform));
        }

        // ============================================================================
        // Matrix construction helpers
        // ============================================================================

        // Internal implementations for matrix construction
        namespace Detail {
            [[nodiscard]] inline IR::Value::Expr<Math::Mat2> MakeMat2Impl(
                const IR::Value::Expr<Math::Vec2>& c0,
                const IR::Value::Expr<Math::Vec2>& c1) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c1)
                };
                return Math::MakeCall<Math::Mat2>("mat2", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat3> MakeMat3Impl(
                const IR::Value::Expr<Math::Vec3>& c0,
                const IR::Value::Expr<Math::Vec3>& c1,
                const IR::Value::Expr<Math::Vec3>& c2) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c2)
                };
                return Math::MakeCall<Math::Mat3>("mat3", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat4> MakeMat4Impl(
                const IR::Value::Expr<Math::Vec4>& c0,
                const IR::Value::Expr<Math::Vec4>& c1,
                const IR::Value::Expr<Math::Vec4>& c2,
                const IR::Value::Expr<Math::Vec4>& c3) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c2),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c3)
                };
                return Math::MakeCall<Math::Mat4>("mat4", BuildVectorParams(exprs));
            }

            // Rectangular matrices
            [[nodiscard]] inline IR::Value::Expr<Math::Mat2x3> MakeMat2x3Impl(
                const IR::Value::Expr<Math::Vec3>& c0,
                const IR::Value::Expr<Math::Vec3>& c1) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c1)
                };
                return Math::MakeCall<Math::Mat2x3>("mat2x3", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat3x2> MakeMat3x2Impl(
                const IR::Value::Expr<Math::Vec2>& c0,
                const IR::Value::Expr<Math::Vec2>& c1,
                const IR::Value::Expr<Math::Vec2>& c2) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c2)
                };
                return Math::MakeCall<Math::Mat3x2>("mat3x2", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat2x4> MakeMat2x4Impl(
                const IR::Value::Expr<Math::Vec4>& c0,
                const IR::Value::Expr<Math::Vec4>& c1) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c1)
                };
                return Math::MakeCall<Math::Mat2x4>("mat2x4", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat4x2> MakeMat4x2Impl(
                const IR::Value::Expr<Math::Vec2>& c0,
                const IR::Value::Expr<Math::Vec2>& c1,
                const IR::Value::Expr<Math::Vec2>& c2,
                const IR::Value::Expr<Math::Vec2>& c3) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c2),
                    const_cast<IR::Value::Expr<Math::Vec2>*>(&c3)
                };
                return Math::MakeCall<Math::Mat4x2>("mat4x2", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat3x4> MakeMat3x4Impl(
                const IR::Value::Expr<Math::Vec4>& c0,
                const IR::Value::Expr<Math::Vec4>& c1,
                const IR::Value::Expr<Math::Vec4>& c2) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec4>*>(&c2)
                };
                return Math::MakeCall<Math::Mat3x4>("mat3x4", BuildVectorParams(exprs));
            }

            [[nodiscard]] inline IR::Value::Expr<Math::Mat4x3> MakeMat4x3Impl(
                const IR::Value::Expr<Math::Vec3>& c0,
                const IR::Value::Expr<Math::Vec3>& c1,
                const IR::Value::Expr<Math::Vec3>& c2,
                const IR::Value::Expr<Math::Vec3>& c3) {
                std::vector<IR::Value::ExprBase*> exprs = {
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c0),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c1),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c2),
                    const_cast<IR::Value::Expr<Math::Vec3>*>(&c3)
                };
                return Math::MakeCall<Math::Mat4x3>("mat4x3", BuildVectorParams(exprs));
            }
        }

        // Public template versions for matrix construction from columns with type constraints
        template<typename C0, typename C1>
            requires Detail::Vec2Component<C0> && Detail::Vec2Component<C1>
        [[nodiscard]] inline auto MakeMat2(C0&& c0, C1&& c1) {
            return Detail::MakeMat2Impl(
                IR::Value::Expr<Math::Vec2>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec2>(std::forward<C1>(c1)));
        }

        template<typename C0, typename C1, typename C2>
            requires Detail::Vec3Component<C0> && Detail::Vec3Component<C1> && Detail::Vec3Component<C2>
        [[nodiscard]] inline auto MakeMat3(C0&& c0, C1&& c1, C2&& c2) {
            return Detail::MakeMat3Impl(
                IR::Value::Expr<Math::Vec3>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec3>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec3>(std::forward<C2>(c2)));
        }

        template<typename C0, typename C1, typename C2, typename C3>
            requires Detail::Vec4Component<C0> && Detail::Vec4Component<C1> && 
                     Detail::Vec4Component<C2> && Detail::Vec4Component<C3>
        [[nodiscard]] inline auto MakeMat4(C0&& c0, C1&& c1, C2&& c2, C3&& c3) {
            return Detail::MakeMat4Impl(
                IR::Value::Expr<Math::Vec4>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec4>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec4>(std::forward<C2>(c2)),
                IR::Value::Expr<Math::Vec4>(std::forward<C3>(c3)));
        }

        // Rectangular matrices
        template<typename C0, typename C1>
            requires Detail::Vec3Component<C0> && Detail::Vec3Component<C1>
        [[nodiscard]] inline auto MakeMat2x3(C0&& c0, C1&& c1) {
            return Detail::MakeMat2x3Impl(
                IR::Value::Expr<Math::Vec3>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec3>(std::forward<C1>(c1)));
        }

        template<typename C0, typename C1, typename C2>
            requires Detail::Vec2Component<C0> && Detail::Vec2Component<C1> && Detail::Vec2Component<C2>
        [[nodiscard]] inline auto MakeMat3x2(C0&& c0, C1&& c1, C2&& c2) {
            return Detail::MakeMat3x2Impl(
                IR::Value::Expr<Math::Vec2>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec2>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec2>(std::forward<C2>(c2)));
        }

        template<typename C0, typename C1>
            requires Detail::Vec4Component<C0> && Detail::Vec4Component<C1>
        [[nodiscard]] inline auto MakeMat2x4(C0&& c0, C1&& c1) {
            return Detail::MakeMat2x4Impl(
                IR::Value::Expr<Math::Vec4>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec4>(std::forward<C1>(c1)));
        }

        template<typename C0, typename C1, typename C2, typename C3>
            requires Detail::Vec2Component<C0> && Detail::Vec2Component<C1> && 
                     Detail::Vec2Component<C2> && Detail::Vec2Component<C3>
        [[nodiscard]] inline auto MakeMat4x2(C0&& c0, C1&& c1, C2&& c2, C3&& c3) {
            return Detail::MakeMat4x2Impl(
                IR::Value::Expr<Math::Vec2>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec2>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec2>(std::forward<C2>(c2)),
                IR::Value::Expr<Math::Vec2>(std::forward<C3>(c3)));
        }

        template<typename C0, typename C1, typename C2>
            requires Detail::Vec4Component<C0> && Detail::Vec4Component<C1> && Detail::Vec4Component<C2>
        [[nodiscard]] inline auto MakeMat3x4(C0&& c0, C1&& c1, C2&& c2) {
            return Detail::MakeMat3x4Impl(
                IR::Value::Expr<Math::Vec4>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec4>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec4>(std::forward<C2>(c2)));
        }

        template<typename C0, typename C1, typename C2, typename C3>
            requires Detail::Vec3Component<C0> && Detail::Vec3Component<C1> && 
                     Detail::Vec3Component<C2> && Detail::Vec3Component<C3>
        [[nodiscard]] inline auto MakeMat4x3(C0&& c0, C1&& c1, C2&& c2, C3&& c3) {
            return Detail::MakeMat4x3Impl(
                IR::Value::Expr<Math::Vec3>(std::forward<C0>(c0)),
                IR::Value::Expr<Math::Vec3>(std::forward<C1>(c1)),
                IR::Value::Expr<Math::Vec3>(std::forward<C2>(c2)),
                IR::Value::Expr<Math::Vec3>(std::forward<C3>(c3)));
        }

        // CPU-side matrix to GPU Expr matrix conversions
        [[nodiscard]] inline IR::Value::Expr<Math::Mat2> MakeMat2(const Math::Mat2& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat2>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat3> MakeMat3(const Math::Mat3& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat3>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat4> MakeMat4(const Math::Mat4& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat4>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat2x3> MakeMat2x3(const Math::Mat2x3& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat2x3>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat3x2> MakeMat3x2(const Math::Mat3x2& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat3x2>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat2x4> MakeMat2x4(const Math::Mat2x4& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat2x4>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat4x2> MakeMat4x2(const Math::Mat4x2& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat4x2>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat3x4> MakeMat3x4(const Math::Mat3x4& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat3x4>(std::move(uniform));
        }

        [[nodiscard]] inline IR::Value::Expr<Math::Mat4x3> MakeMat4x3(const Math::Mat4x3& m) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m));
            return IR::Value::Expr<Math::Mat4x3>(std::move(uniform));
        }
    }

    // ============================================================================
    // Scalar construction helpers
    // ============================================================================
    
    // Trait to detect if T is Var<U> for some U
    template<typename T>
    struct IsVar : std::false_type {};
    template<typename U>
    struct IsVar<IR::Value::Var<U>> : std::true_type {};

    // Trait to detect if T is Expr<U> for some U
    template<typename T>
    struct IsExpr : std::false_type {};
    template<typename U>
    struct IsExpr<IR::Value::Expr<U>> : std::true_type {};

    // Trait to get value type from Var<T> or Expr<T>
    template<typename T>
    struct ValueTypeOf {
        using type = void;
    };
    template<typename U>
    struct ValueTypeOf<IR::Value::Var<U>> {
        using type = U;
    };
    template<typename U>
    struct ValueTypeOf<IR::Value::Expr<U>> {
        using type = U;
    };
    template<typename T>
    using ValueTypeOf_t = typename ValueTypeOf<std::remove_cvref_t<T>>::type;

    // Helper concepts to avoid Intellisense warnings
    template<typename T, typename U>
    concept IsVarOfType = IsVar<std::remove_cvref_t<T>>::value && std::same_as<ValueTypeOf_t<T>, U>;

    template<typename T, typename U>
    concept IsExprOfType = IsExpr<std::remove_cvref_t<T>>::value && std::same_as<ValueTypeOf_t<T>, U>;

    // Concept for valid float type (Var<float>, Expr<float>, or convertible-to-float literal)
    template<typename T>
    concept FloatConstructible = 
        std::convertible_to<std::remove_cvref_t<T>, float> ||
        IsVarOfType<T, float> ||
        IsExprOfType<T, float>;

    // Concept for valid int type (Var<int>, Expr<int>, or convertible-to-int literal)
    template<typename T>
    concept IntConstructible = 
        std::convertible_to<std::remove_cvref_t<T>, int> ||
        IsVarOfType<T, int> ||
        IsExprOfType<T, int>;

    // Concept for valid bool type (Var<bool>, Expr<bool>, or convertible-to-bool literal)
    template<typename T>
    concept BoolConstructible = 
        std::convertible_to<std::remove_cvref_t<T>, bool> ||
        IsVarOfType<T, bool> ||
        IsExprOfType<T, bool>;

    // MakeFloat - Construct Expr<float> from scalar values
    template<typename T>
        requires FloatConstructible<T>
    [[nodiscard]] inline auto MakeFloat(T&& x) {
        return IR::Value::Expr<float>(std::forward<T>(x));
    }

    // MakeInt - Construct Expr<int> from scalar values  
    template<typename T>
        requires IntConstructible<T>
    [[nodiscard]] inline auto MakeInt(T&& x) {
        return IR::Value::Expr<int>(std::forward<T>(x));
    }

    // MakeBool - Construct Expr<bool> from scalar values
    template<typename T>
        requires BoolConstructible<T>
    [[nodiscard]] inline auto MakeBool(T&& x) {
        return IR::Value::Expr<bool>(std::forward<T>(x));
    }

    // ============================================================================
    // Convenience: bring Alias and Construct into GPU namespace
    // ============================================================================
    using namespace Alias;
    using namespace Construct;
}

#endif //EASYGPU_HELPERS_H
