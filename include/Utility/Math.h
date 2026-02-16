#pragma once

/**
 * Math.h:
 *      @Descripiton    :   The math library for GPU programing
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_MATH_H
#define EASYGPU_MATH_H

#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>

#include <IR/Node/Node.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/LoadUniform.h>

#include <vector>
#include <memory>
#include <string>
#include <concepts>

namespace GPU::Math {
    // ============================================================================
    // Helper functions for building intrinsic calls
    // ============================================================================
    
    /**
     * Making a system function call
     * @tparam T The type of the return value
     * @param Name The name of the calling function
     * @param Parameters The parameters of the call
     * @return The expression which contains a function calling
     */
    template<typename T>
    IR::Value::Expr<T> MakeCall(const std::string& Name, std::vector<std::unique_ptr<IR::Node::Node*>> Parameters) {
        auto callNode = std::make_unique<IR::Node::IntrinsicCallNode>(Name, std::move(Parameters));
        return IR::Value::Expr<T>(std::move(callNode));
    }

    /**
     * Helper to wrap a single Expr parameter
     */
    template<typename T>
    std::unique_ptr<IR::Node::Node*> WrapNode(const IR::Value::Expr<T>& expr) {
        return std::make_unique<IR::Node::Node*>(expr.Node()->Clone().release());
    }

    /**
     * Helper to wrap an ExprBase parameter (for bool expressions)
     */
    inline std::unique_ptr<IR::Node::Node*> WrapNode(const IR::Value::ExprBase& expr) {
        return std::make_unique<IR::Node::Node*>(expr.Node()->Clone().release());
    }

    // ============================================================================
    // Common Helper for building parameter lists
    // ============================================================================
    
    namespace Detail {
        // Helper to check if a type is a scalar (not an Expr or Var)
        template<typename T>
        struct IsScalar {
            static constexpr bool value = !requires { typename T::ValueType; } && 
                                          (std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>);
        };
        
        // Type trait to get the value type (element type) from Expr<T> or Var<T>
        template<typename T>
        struct ValueTypeOf {
            using type = T;
        };
        template<typename T>
        struct ValueTypeOf<IR::Value::Expr<T>> {
            using type = T;
        };
        template<typename T>
        struct ValueTypeOf<IR::Value::Var<T>> {
            using type = T;
        };
        template<typename T>
        using ValueTypeOf_t = typename ValueTypeOf<std::remove_cvref_t<T>>::type;
        
        // Trait to detect if T is Var<...> (has conversion to Expr but is not Expr itself)
        template<typename T>
        struct IsVar : std::false_type {};
        template<typename T>
        struct IsVar<IR::Value::Var<T>> : std::true_type {};
        
        // Trait to detect if T is a CPU-side Vec type (Math::Vec2/3/4, IVec2/3/4)
        template<typename T>
        struct IsCpuVec : std::false_type {};
        template<> struct IsCpuVec<Math::Vec2> : std::true_type {};
        template<> struct IsCpuVec<Math::Vec3> : std::true_type {};
        template<> struct IsCpuVec<Math::Vec4> : std::true_type {};
        template<> struct IsCpuVec<Math::IVec2> : std::true_type {};
        template<> struct IsCpuVec<Math::IVec3> : std::true_type {};
        template<> struct IsCpuVec<Math::IVec4> : std::true_type {};
        
        // Convert any argument (Expr, Var, CPU Vec, or scalar) to ExprBase (for use in AddParam)
        template<typename T>
        [[nodiscard]] inline auto ToExpr(T&& val) -> IR::Value::ExprBase {
            using U = std::remove_cvref_t<T>;
            using ValueType = ValueTypeOf_t<T>;
            
            if constexpr (IsVar<U>::value) {
                // Var<T> - convert to Expr
                return IR::Value::Expr<ValueType>(std::forward<T>(val));
            } else if constexpr (requires { typename U::ValueType; }) {
                // Expr<T> - release and reconstruct
                // Expr is not movable in specializations, so we release and reconstruct via ExprBase
                return IR::Value::ExprBase(std::forward<T>(val).Release());
            } else if constexpr (IsCpuVec<U>::value) {
                // CPU-side Vec - construct via ValueToString (creates uniform)
                auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString<U>(val));
                return IR::Value::ExprBase(std::move(uniform));
            } else {
                // Scalar - construct Expr
                return IR::Value::Expr<ValueType>(std::forward<T>(val));
            }
        }
        
        // AddParam for Expr types - wrap the node
        template<typename T>
        void AddParam(std::vector<std::unique_ptr<IR::Node::Node*>>& params, const IR::Value::Expr<T>& arg) {
            params.push_back(WrapNode(arg));
        }
        
        // AddParam for ExprBase - direct wrap (used by ToExpr)
        inline void AddParam(std::vector<std::unique_ptr<IR::Node::Node*>>& params, const IR::Value::ExprBase& arg) {
            params.push_back(std::make_unique<IR::Node::Node*>(const_cast<IR::Value::ExprBase&>(arg).Node()->Clone().release()));
        }
        
        // AddParam for scalar types - convert to uniform load
        template<typename T>
            requires IsScalar<T>::value
        void AddParam(std::vector<std::unique_ptr<IR::Node::Node*>>& params, T arg) {
            auto uniform = std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString<T>(arg));
            params.push_back(std::make_unique<IR::Node::Node*>(uniform.release()));
        }
        
        // Base case: single parameter
        template<typename T1>
        std::vector<std::unique_ptr<IR::Node::Node*>> BuildParams(T1&& arg1) {
            std::vector<std::unique_ptr<IR::Node::Node*>> params;
            AddParam(params, std::forward<T1>(arg1));
            return params;
        }
        
        // Two parameters
        template<typename T1, typename T2>
        std::vector<std::unique_ptr<IR::Node::Node*>> BuildParams(T1&& arg1, T2&& arg2) {
            std::vector<std::unique_ptr<IR::Node::Node*>> params;
            AddParam(params, std::forward<T1>(arg1));
            AddParam(params, std::forward<T2>(arg2));
            return params;
        }
        
        // Three parameters
        template<typename T1, typename T2, typename T3>
        std::vector<std::unique_ptr<IR::Node::Node*>> BuildParams(T1&& arg1, T2&& arg2, T3&& arg3) {
            std::vector<std::unique_ptr<IR::Node::Node*>> params;
            AddParam(params, std::forward<T1>(arg1));
            AddParam(params, std::forward<T2>(arg2));
            AddParam(params, std::forward<T3>(arg3));
            return params;
        }
        
        // Template helper to create math functions with automatic type conversion
        template<typename T>
        using expr_t = IR::Value::Expr<ValueTypeOf_t<T>>;
    }

    // ============================================================================
    // Trigonometric Functions
    // ============================================================================
    
    // Sin
    inline IR::Value::Expr<float> Sin(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("sin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Sin(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("sin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Sin(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("sin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Sin(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("sin", Detail::BuildParams(x));
    }
    
    // Cos
    inline IR::Value::Expr<float> Cos(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("cos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Cos(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("cos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Cos(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("cos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Cos(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("cos", Detail::BuildParams(x));
    }
    
    // Tan
    inline IR::Value::Expr<float> Tan(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("tan", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Tan(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("tan", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Tan(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("tan", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Tan(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("tan", Detail::BuildParams(x));
    }
    
    // Asin
    inline IR::Value::Expr<float> Asin(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("asin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Asin(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("asin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Asin(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("asin", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Asin(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("asin", Detail::BuildParams(x));
    }
    
    // Acos
    inline IR::Value::Expr<float> Acos(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("acos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Acos(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("acos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Acos(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("acos", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Acos(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("acos", Detail::BuildParams(x));
    }
    
    // Atan
    inline IR::Value::Expr<float> Atan(const IR::Value::Expr<float>& y_over_x) {
        return MakeCall<float>("atan", Detail::BuildParams(y_over_x));
    }
    inline IR::Value::Expr<Vec2> Atan(const IR::Value::Expr<Vec2>& y_over_x) {
        return MakeCall<Vec2>("atan", Detail::BuildParams(y_over_x));
    }
    inline IR::Value::Expr<Vec3> Atan(const IR::Value::Expr<Vec3>& y_over_x) {
        return MakeCall<Vec3>("atan", Detail::BuildParams(y_over_x));
    }
    inline IR::Value::Expr<Vec4> Atan(const IR::Value::Expr<Vec4>& y_over_x) {
        return MakeCall<Vec4>("atan", Detail::BuildParams(y_over_x));
    }
    
    // Atan2
    inline IR::Value::Expr<float> Atan2(const IR::Value::Expr<float>& y, const IR::Value::Expr<float>& x) {
        return MakeCall<float>("atan", Detail::BuildParams(y, x));
    }
    inline IR::Value::Expr<Vec2> Atan2(const IR::Value::Expr<Vec2>& y, const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("atan", Detail::BuildParams(y, x));
    }
    inline IR::Value::Expr<Vec3> Atan2(const IR::Value::Expr<Vec3>& y, const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("atan", Detail::BuildParams(y, x));
    }
    inline IR::Value::Expr<Vec4> Atan2(const IR::Value::Expr<Vec4>& y, const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("atan", Detail::BuildParams(y, x));
    }
    
    // Sinh
    inline IR::Value::Expr<float> Sinh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("sinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Sinh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("sinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Sinh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("sinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Sinh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("sinh", Detail::BuildParams(x));
    }
    
    // Cosh
    inline IR::Value::Expr<float> Cosh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("cosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Cosh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("cosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Cosh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("cosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Cosh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("cosh", Detail::BuildParams(x));
    }
    
    // Tanh
    inline IR::Value::Expr<float> Tanh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("tanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Tanh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("tanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Tanh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("tanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Tanh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("tanh", Detail::BuildParams(x));
    }
    
    // Asinh
    inline IR::Value::Expr<float> Asinh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("asinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Asinh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("asinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Asinh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("asinh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Asinh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("asinh", Detail::BuildParams(x));
    }
    
    // Acosh
    inline IR::Value::Expr<float> Acosh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("acosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Acosh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("acosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Acosh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("acosh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Acosh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("acosh", Detail::BuildParams(x));
    }
    
    // Atanh
    inline IR::Value::Expr<float> Atanh(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("atanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Atanh(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("atanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Atanh(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("atanh", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Atanh(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("atanh", Detail::BuildParams(x));
    }
    
    // Radians (Degrees to Radians)
    inline IR::Value::Expr<float> Radians(const IR::Value::Expr<float>& Degrees) {
        return MakeCall<float>("radians", Detail::BuildParams(Degrees));
    }
    inline IR::Value::Expr<Vec2> Radians(const IR::Value::Expr<Vec2>& Degrees) {
        return MakeCall<Vec2>("radians", Detail::BuildParams(Degrees));
    }
    inline IR::Value::Expr<Vec3> Radians(const IR::Value::Expr<Vec3>& Degrees) {
        return MakeCall<Vec3>("radians", Detail::BuildParams(Degrees));
    }
    inline IR::Value::Expr<Vec4> Radians(const IR::Value::Expr<Vec4>& Degrees) {
        return MakeCall<Vec4>("radians", Detail::BuildParams(Degrees));
    }
    
    // Degrees (Radians to Degrees)
    inline IR::Value::Expr<float> Degrees(const IR::Value::Expr<float>& Radians) {
        return MakeCall<float>("degrees", Detail::BuildParams(Radians));
    }
    inline IR::Value::Expr<Vec2> Degrees(const IR::Value::Expr<Vec2>& Radians) {
        return MakeCall<Vec2>("degrees", Detail::BuildParams(Radians));
    }
    inline IR::Value::Expr<Vec3> Degrees(const IR::Value::Expr<Vec3>& Radians) {
        return MakeCall<Vec3>("degrees", Detail::BuildParams(Radians));
    }
    inline IR::Value::Expr<Vec4> Degrees(const IR::Value::Expr<Vec4>& Radians) {
        return MakeCall<Vec4>("degrees", Detail::BuildParams(Radians));
    }

    // ============================================================================
    // Exponential Functions
    // ============================================================================
    
    // Pow
    inline IR::Value::Expr<float> Pow(const IR::Value::Expr<float>& base, const IR::Value::Expr<float>& Exp) {
        return MakeCall<float>("pow", Detail::BuildParams(base, Exp));
    }
    inline IR::Value::Expr<Vec2> Pow(const IR::Value::Expr<Vec2>& base, const IR::Value::Expr<Vec2>& Exp) {
        return MakeCall<Vec2>("pow", Detail::BuildParams(base, Exp));
    }
    inline IR::Value::Expr<Vec3> Pow(const IR::Value::Expr<Vec3>& base, const IR::Value::Expr<Vec3>& Exp) {
        return MakeCall<Vec3>("pow", Detail::BuildParams(base, Exp));
    }
    inline IR::Value::Expr<Vec4> Pow(const IR::Value::Expr<Vec4>& base, const IR::Value::Expr<Vec4>& Exp) {
        return MakeCall<Vec4>("pow", Detail::BuildParams(base, Exp));
    }
    
    // Pow - Generic template versions
    template<typename Base, typename Exp>
    [[nodiscard]] inline auto Pow(Base&& base, Exp&& exp) {
        using T = Detail::ValueTypeOf_t<Base>;
        return MakeCall<T>("pow", Detail::BuildParams(
            Detail::ToExpr(std::forward<Base>(base)),
            Detail::ToExpr(std::forward<Exp>(exp))
        ));
    }
    
    // Exp
    inline IR::Value::Expr<float> Exp(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("exp", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Exp(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("exp", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Exp(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("exp", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Exp(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("exp", Detail::BuildParams(x));
    }
    
    // Log
    inline IR::Value::Expr<float> Log(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("log", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Log(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("log", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Log(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("log", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Log(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("log", Detail::BuildParams(x));
    }
    
    // Exp2
    inline IR::Value::Expr<float> Exp2(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("exp2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Exp2(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("exp2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Exp2(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("exp2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Exp2(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("exp2", Detail::BuildParams(x));
    }
    
    // Log2
    inline IR::Value::Expr<float> Log2(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("log2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Log2(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("log2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Log2(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("log2", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Log2(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("log2", Detail::BuildParams(x));
    }
    
    // Sqrt
    inline IR::Value::Expr<float> Sqrt(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("sqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Sqrt(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("sqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Sqrt(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("sqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Sqrt(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("sqrt", Detail::BuildParams(x));
    }
    
    // Inversesqrt
    inline IR::Value::Expr<float> Inversesqrt(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("inversesqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Inversesqrt(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("inversesqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Inversesqrt(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("inversesqrt", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Inversesqrt(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("inversesqrt", Detail::BuildParams(x));
    }

    // ============================================================================
    // Common Functions
    // ============================================================================
    
    // Abs
    inline IR::Value::Expr<float> Abs(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Abs(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Abs(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Abs(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<int> Abs(const IR::Value::Expr<int>& x) {
        return MakeCall<int>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec2> Abs(const IR::Value::Expr<IVec2>& x) {
        return MakeCall<IVec2>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec3> Abs(const IR::Value::Expr<IVec3>& x) {
        return MakeCall<IVec3>("abs", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec4> Abs(const IR::Value::Expr<IVec4>& x) {
        return MakeCall<IVec4>("abs", Detail::BuildParams(x));
    }
    
    // Sign
    inline IR::Value::Expr<float> Sign(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Sign(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Sign(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Sign(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<int> Sign(const IR::Value::Expr<int>& x) {
        return MakeCall<int>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec2> Sign(const IR::Value::Expr<IVec2>& x) {
        return MakeCall<IVec2>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec3> Sign(const IR::Value::Expr<IVec3>& x) {
        return MakeCall<IVec3>("sign", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<IVec4> Sign(const IR::Value::Expr<IVec4>& x) {
        return MakeCall<IVec4>("sign", Detail::BuildParams(x));
    }
    
    // Floor
    inline IR::Value::Expr<float> Floor(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("floor", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Floor(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("floor", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Floor(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("floor", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Floor(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("floor", Detail::BuildParams(x));
    }
    
    // Trunc
    inline IR::Value::Expr<float> Trunc(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("trunc", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Trunc(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("trunc", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Trunc(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("trunc", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Trunc(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("trunc", Detail::BuildParams(x));
    }
    
    // Round
    inline IR::Value::Expr<float> Round(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("round", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Round(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("round", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Round(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("round", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Round(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("round", Detail::BuildParams(x));
    }
    
    // RoundEven
    inline IR::Value::Expr<float> RoundEven(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("roundEven", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> RoundEven(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("roundEven", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> RoundEven(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("roundEven", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> RoundEven(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("roundEven", Detail::BuildParams(x));
    }
    
    // Ceil
    inline IR::Value::Expr<float> Ceil(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("ceil", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Ceil(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("ceil", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Ceil(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("ceil", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Ceil(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("ceil", Detail::BuildParams(x));
    }
    
    // Fract
    inline IR::Value::Expr<float> Fract(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("fract", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Fract(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("fract", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Fract(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("fract", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Fract(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("fract", Detail::BuildParams(x));
    }
    
    // Mod
    inline IR::Value::Expr<float> Mod(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y) {
        return MakeCall<float>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Mod(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<Vec2>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Mod(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<Vec3>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Mod(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<Vec4>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Mod(const IR::Value::Expr<float>& x, float y) {
        return MakeCall<float>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Mod(const IR::Value::Expr<Vec2>& x, float y) {
        return MakeCall<Vec2>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Mod(const IR::Value::Expr<Vec3>& x, float y) {
        return MakeCall<Vec3>("mod", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Mod(const IR::Value::Expr<Vec4>& x, float y) {
        return MakeCall<Vec4>("mod", Detail::BuildParams(x, y));
    }
    
    // Mod - Generic template versions
    template<typename X, typename Y>
    [[nodiscard]] inline auto Mod(X&& x, Y&& y) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("mod", Detail::BuildParams(
            Detail::ToExpr(std::forward<X>(x)),
            Detail::ToExpr(std::forward<Y>(y))
        ));
    }
    
    // Min
    inline IR::Value::Expr<float> Min(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y) {
        return MakeCall<float>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Min(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<Vec2>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Min(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<Vec3>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Min(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<Vec4>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Min(const IR::Value::Expr<float>& x, float y) {
        return MakeCall<float>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Min(const IR::Value::Expr<Vec2>& x, float y) {
        return MakeCall<Vec2>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Min(const IR::Value::Expr<Vec3>& x, float y) {
        return MakeCall<Vec3>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Min(const IR::Value::Expr<Vec4>& x, float y) {
        return MakeCall<Vec4>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Min(const IR::Value::Expr<int>& x, const IR::Value::Expr<int>& y) {
        return MakeCall<int>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Min(const IR::Value::Expr<int>& x, int y) {
        return MakeCall<int>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Min(int x, const IR::Value::Expr<int>& y) {
        return MakeCall<int>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec2> Min(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<IVec2>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec3> Min(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<IVec3>("min", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec4> Min(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<IVec4>("min", Detail::BuildParams(x, y));
    }
    
    // Max
    inline IR::Value::Expr<float> Max(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y) {
        return MakeCall<float>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Max(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<Vec2>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Max(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<Vec3>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Max(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<Vec4>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Max(const IR::Value::Expr<float>& x, float y) {
        return MakeCall<float>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec2> Max(const IR::Value::Expr<Vec2>& x, float y) {
        return MakeCall<Vec2>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec3> Max(const IR::Value::Expr<Vec3>& x, float y) {
        return MakeCall<Vec3>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<Vec4> Max(const IR::Value::Expr<Vec4>& x, float y) {
        return MakeCall<Vec4>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Max(const IR::Value::Expr<int>& x, const IR::Value::Expr<int>& y) {
        return MakeCall<int>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Max(const IR::Value::Expr<int>& x, int y) {
        return MakeCall<int>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<int> Max(int x, const IR::Value::Expr<int>& y) {
        return MakeCall<int>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec2> Max(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<IVec2>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec3> Max(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<IVec3>("max", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<IVec4> Max(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<IVec4>("max", Detail::BuildParams(x, y));
    }
    
    // Min - Generic template versions
    template<typename X, typename Y>
    [[nodiscard]] inline auto Min(X&& x, Y&& y) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("min", Detail::BuildParams(
            Detail::ToExpr(std::forward<X>(x)),
            Detail::ToExpr(std::forward<Y>(y))
        ));
    }
    
    // Max - Generic template versions  
    template<typename X, typename Y>
    [[nodiscard]] inline auto Max(X&& x, Y&& y) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("max", Detail::BuildParams(
            Detail::ToExpr(std::forward<X>(x)),
            Detail::ToExpr(std::forward<Y>(y))
        ));
    }
    
    // Clamp
    inline IR::Value::Expr<float> Clamp(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& minVal, const IR::Value::Expr<float>& maxVal) {
        return MakeCall<float>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec2> Clamp(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& minVal, const IR::Value::Expr<Vec2>& maxVal) {
        return MakeCall<Vec2>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec3> Clamp(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& minVal, const IR::Value::Expr<Vec3>& maxVal) {
        return MakeCall<Vec3>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec4> Clamp(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& minVal, const IR::Value::Expr<Vec4>& maxVal) {
        return MakeCall<Vec4>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<float> Clamp(const IR::Value::Expr<float>& x, float minVal, float maxVal) {
        return MakeCall<float>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec2> Clamp(const IR::Value::Expr<Vec2>& x, float minVal, float maxVal) {
        return MakeCall<Vec2>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec3> Clamp(const IR::Value::Expr<Vec3>& x, float minVal, float maxVal) {
        return MakeCall<Vec3>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<Vec4> Clamp(const IR::Value::Expr<Vec4>& x, float minVal, float maxVal) {
        return MakeCall<Vec4>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<int> Clamp(const IR::Value::Expr<int>& x, const IR::Value::Expr<int>& minVal, const IR::Value::Expr<int>& maxVal) {
        return MakeCall<int>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<IVec2> Clamp(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& minVal, const IR::Value::Expr<IVec2>& maxVal) {
        return MakeCall<IVec2>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<IVec3> Clamp(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& minVal, const IR::Value::Expr<IVec3>& maxVal) {
        return MakeCall<IVec3>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    inline IR::Value::Expr<IVec4> Clamp(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& minVal, const IR::Value::Expr<IVec4>& maxVal) {
        return MakeCall<IVec4>("clamp", Detail::BuildParams(x, minVal, maxVal));
    }
    
    // Clamp - Generic template versions
    template<typename X, typename Min, typename Max>
    [[nodiscard]] inline auto Clamp(X&& x, Min&& minVal, Max&& maxVal) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("clamp", Detail::BuildParams(
            Detail::ToExpr(std::forward<X>(x)),
            Detail::ToExpr(std::forward<Min>(minVal)),
            Detail::ToExpr(std::forward<Max>(maxVal))
        ));
    }
    
    // Mix (lerp) - Expr versions
    inline IR::Value::Expr<float> Mix(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y, const IR::Value::Expr<float>& a) {
        return MakeCall<float>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec2> Mix(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y, const IR::Value::Expr<Vec2>& a) {
        return MakeCall<Vec2>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec3> Mix(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y, const IR::Value::Expr<Vec3>& a) {
        return MakeCall<Vec3>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec4> Mix(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y, const IR::Value::Expr<Vec4>& a) {
        return MakeCall<Vec4>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<float> Mix(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y, float a) {
        return MakeCall<float>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec2> Mix(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y, float a) {
        return MakeCall<Vec2>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec3> Mix(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y, float a) {
        return MakeCall<Vec3>("mix", Detail::BuildParams(x, y, a));
    }
    inline IR::Value::Expr<Vec4> Mix(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y, float a) {
        return MakeCall<Vec4>("mix", Detail::BuildParams(x, y, a));
    }
    
    // Mix (lerp) - Generic template versions accepting Expr/Var/scalar combinations
    template<typename X, typename Y, typename A>
    [[nodiscard]] inline auto Mix(X&& x, Y&& y, A&& a) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("mix", Detail::BuildParams(
            Detail::ToExpr(std::forward<X>(x)),
            Detail::ToExpr(std::forward<Y>(y)),
            Detail::ToExpr(std::forward<A>(a))
        ));
    }
    
    // Step
    inline IR::Value::Expr<float> Step(const IR::Value::Expr<float>& edge, const IR::Value::Expr<float>& x) {
        return MakeCall<float>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec2> Step(const IR::Value::Expr<Vec2>& edge, const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec3> Step(const IR::Value::Expr<Vec3>& edge, const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec4> Step(const IR::Value::Expr<Vec4>& edge, const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<float> Step(float edge, const IR::Value::Expr<float>& x) {
        return MakeCall<float>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec2> Step(float edge, const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec3> Step(float edge, const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("step", Detail::BuildParams(edge, x));
    }
    inline IR::Value::Expr<Vec4> Step(float edge, const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("step", Detail::BuildParams(edge, x));
    }
    
    // Step - Generic template versions
    template<typename Edge, typename X>
    [[nodiscard]] inline auto Step(Edge&& edge, X&& x) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("step", Detail::BuildParams(
            Detail::ToExpr(std::forward<Edge>(edge)),
            Detail::ToExpr(std::forward<X>(x))
        ));
    }
    
    // Smoothstep
    inline IR::Value::Expr<float> Smoothstep(const IR::Value::Expr<float>& edge0, const IR::Value::Expr<float>& edge1, const IR::Value::Expr<float>& x) {
        return MakeCall<float>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec2> Smoothstep(const IR::Value::Expr<Vec2>& edge0, const IR::Value::Expr<Vec2>& edge1, const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec3> Smoothstep(const IR::Value::Expr<Vec3>& edge0, const IR::Value::Expr<Vec3>& edge1, const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec4> Smoothstep(const IR::Value::Expr<Vec4>& edge0, const IR::Value::Expr<Vec4>& edge1, const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<float> Smoothstep(float edge0, float edge1, const IR::Value::Expr<float>& x) {
        return MakeCall<float>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec2> Smoothstep(float edge0, float edge1, const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec3> Smoothstep(float edge0, float edge1, const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    inline IR::Value::Expr<Vec4> Smoothstep(float edge0, float edge1, const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("smoothstep", Detail::BuildParams(edge0, edge1, x));
    }
    
    // Smoothstep - Generic template versions
    template<typename E0, typename E1, typename X>
    [[nodiscard]] inline auto Smoothstep(E0&& edge0, E1&& edge1, X&& x) {
        using T = Detail::ValueTypeOf_t<X>;
        return MakeCall<T>("smoothstep", Detail::BuildParams(
            Detail::ToExpr(std::forward<E0>(edge0)),
            Detail::ToExpr(std::forward<E1>(edge1)),
            Detail::ToExpr(std::forward<X>(x))
        ));
    }

    // ============================================================================
    // Geometric Functions
    // ============================================================================
    
    // Length
    inline IR::Value::Expr<float> Length(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("length", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<float> Length(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<float>("length", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<float> Length(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<float>("length", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<float> Length(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<float>("length", Detail::BuildParams(x));
    }
    
    // Distance
    inline IR::Value::Expr<float> Distance(const IR::Value::Expr<float>& p0, const IR::Value::Expr<float>& p1) {
        return MakeCall<float>("distance", Detail::BuildParams(p0, p1));
    }
    inline IR::Value::Expr<float> Distance(const IR::Value::Expr<Vec2>& p0, const IR::Value::Expr<Vec2>& p1) {
        return MakeCall<float>("distance", Detail::BuildParams(p0, p1));
    }
    inline IR::Value::Expr<float> Distance(const IR::Value::Expr<Vec3>& p0, const IR::Value::Expr<Vec3>& p1) {
        return MakeCall<float>("distance", Detail::BuildParams(p0, p1));
    }
    inline IR::Value::Expr<float> Distance(const IR::Value::Expr<Vec4>& p0, const IR::Value::Expr<Vec4>& p1) {
        return MakeCall<float>("distance", Detail::BuildParams(p0, p1));
    }
    
    // Dot
    inline IR::Value::Expr<float> Dot(const IR::Value::Expr<float>& x, const IR::Value::Expr<float>& y) {
        return MakeCall<float>("dot", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Dot(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<float>("dot", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Dot(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<float>("dot", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<float> Dot(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<float>("dot", Detail::BuildParams(x, y));
    }
    
    // Cross
    inline IR::Value::Expr<Vec3> Cross(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<Vec3>("cross", Detail::BuildParams(x, y));
    }
    
    // Normalize
    inline IR::Value::Expr<float> Normalize(const IR::Value::Expr<float>& x) {
        return MakeCall<float>("normalize", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec2> Normalize(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<Vec2>("normalize", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec3> Normalize(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<Vec3>("normalize", Detail::BuildParams(x));
    }
    inline IR::Value::Expr<Vec4> Normalize(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<Vec4>("normalize", Detail::BuildParams(x));
    }
    
    // Faceforward
    inline IR::Value::Expr<float> Faceforward(const IR::Value::Expr<float>& N, const IR::Value::Expr<float>& I, const IR::Value::Expr<float>& Nref) {
        return MakeCall<float>("faceforward", Detail::BuildParams(N, I, Nref));
    }
    inline IR::Value::Expr<Vec2> Faceforward(const IR::Value::Expr<Vec2>& N, const IR::Value::Expr<Vec2>& I, const IR::Value::Expr<Vec2>& Nref) {
        return MakeCall<Vec2>("faceforward", Detail::BuildParams(N, I, Nref));
    }
    inline IR::Value::Expr<Vec3> Faceforward(const IR::Value::Expr<Vec3>& N, const IR::Value::Expr<Vec3>& I, const IR::Value::Expr<Vec3>& Nref) {
        return MakeCall<Vec3>("faceforward", Detail::BuildParams(N, I, Nref));
    }
    inline IR::Value::Expr<Vec4> Faceforward(const IR::Value::Expr<Vec4>& N, const IR::Value::Expr<Vec4>& I, const IR::Value::Expr<Vec4>& Nref) {
        return MakeCall<Vec4>("faceforward", Detail::BuildParams(N, I, Nref));
    }
    
    // Reflect
    inline IR::Value::Expr<float> Reflect(const IR::Value::Expr<float>& I, const IR::Value::Expr<float>& N) {
        return MakeCall<float>("reflect", Detail::BuildParams(I, N));
    }
    inline IR::Value::Expr<Vec2> Reflect(const IR::Value::Expr<Vec2>& I, const IR::Value::Expr<Vec2>& N) {
        return MakeCall<Vec2>("reflect", Detail::BuildParams(I, N));
    }
    inline IR::Value::Expr<Vec3> Reflect(const IR::Value::Expr<Vec3>& I, const IR::Value::Expr<Vec3>& N) {
        return MakeCall<Vec3>("reflect", Detail::BuildParams(I, N));
    }
    inline IR::Value::Expr<Vec4> Reflect(const IR::Value::Expr<Vec4>& I, const IR::Value::Expr<Vec4>& N) {
        return MakeCall<Vec4>("reflect", Detail::BuildParams(I, N));
    }
    
    // Refract
    inline IR::Value::Expr<float> Refract(const IR::Value::Expr<float>& I, const IR::Value::Expr<float>& N, const IR::Value::Expr<float>& eta) {
        return MakeCall<float>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec2> Refract(const IR::Value::Expr<Vec2>& I, const IR::Value::Expr<Vec2>& N, const IR::Value::Expr<float>& eta) {
        return MakeCall<Vec2>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec3> Refract(const IR::Value::Expr<Vec3>& I, const IR::Value::Expr<Vec3>& N, const IR::Value::Expr<float>& eta) {
        return MakeCall<Vec3>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec4> Refract(const IR::Value::Expr<Vec4>& I, const IR::Value::Expr<Vec4>& N, const IR::Value::Expr<float>& eta) {
        return MakeCall<Vec4>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<float> Refract(const IR::Value::Expr<float>& I, const IR::Value::Expr<float>& N, float eta) {
        return MakeCall<float>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec2> Refract(const IR::Value::Expr<Vec2>& I, const IR::Value::Expr<Vec2>& N, float eta) {
        return MakeCall<Vec2>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec3> Refract(const IR::Value::Expr<Vec3>& I, const IR::Value::Expr<Vec3>& N, float eta) {
        return MakeCall<Vec3>("refract", Detail::BuildParams(I, N, eta));
    }
    inline IR::Value::Expr<Vec4> Refract(const IR::Value::Expr<Vec4>& I, const IR::Value::Expr<Vec4>& N, float eta) {
        return MakeCall<Vec4>("refract", Detail::BuildParams(I, N, eta));
    }

    // ============================================================================
    // Vector Relational Functions
    // ============================================================================
    
    // LessThan
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThan(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("lessThan", Detail::BuildParams(x, y));
    }
    
    // LessThanEqual
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> LessThanEqual(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("lessThanEqual", Detail::BuildParams(x, y));
    }
    
    // GreaterThan
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThan(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("greaterThan", Detail::BuildParams(x, y));
    }
    
    // GreaterThanEqual
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> GreaterThanEqual(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("greaterThanEqual", Detail::BuildParams(x, y));
    }
    
    // Equal
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> Equal(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("equal", Detail::BuildParams(x, y));
    }
    
    // NotEqual
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<Vec2>& x, const IR::Value::Expr<Vec2>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<Vec3>& x, const IR::Value::Expr<Vec3>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<Vec4>& x, const IR::Value::Expr<Vec4>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<IVec2>& x, const IR::Value::Expr<IVec2>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<IVec3>& x, const IR::Value::Expr<IVec3>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    inline IR::Value::Expr<bool> NotEqual(const IR::Value::Expr<IVec4>& x, const IR::Value::Expr<IVec4>& y) {
        return MakeCall<bool>("notEqual", Detail::BuildParams(x, y));
    }
    
    // ============================================================================
    // Type Cast Functions (Explicit conversion between types)
    // ============================================================================
    
    // -------------------- Scalar Type Casts --------------------
    
    // ToFloat from Var (explicit overloads to avoid ambiguity)
    inline IR::Value::Expr<float> ToFloat(const IR::Value::Var<float>& x) {
        return ToFloat(IR::Value::Expr<float>(x.Load()));  // Already float
    }
    inline IR::Value::Expr<float> ToFloat(const IR::Value::Var<int>& x) {
        return MakeCall<float>("float", Detail::BuildParams(IR::Value::Expr<int>(x.Load())));
    }
    inline IR::Value::Expr<float> ToFloat(const IR::Value::Var<bool>& x) {
        return MakeCall<float>("float", Detail::BuildParams(IR::Value::Expr<bool>(x.Load())));
    }
    
    // ToFloat from Expr
    inline IR::Value::Expr<float> ToFloat(IR::Value::Expr<float>&& x) {
        return std::move(x);  // Already float, no conversion needed
    }
    inline IR::Value::Expr<float> ToFloat(IR::Value::Expr<int>&& x) {
        return MakeCall<float>("float", Detail::BuildParams(std::move(x)));
    }
    inline IR::Value::Expr<float> ToFloat(IR::Value::Expr<bool>&& x) {
        return MakeCall<float>("float", Detail::BuildParams(std::move(x)));
    }
    
    // ToInt from Var (explicit overloads to avoid ambiguity)
    inline IR::Value::Expr<int> ToInt(const IR::Value::Var<int>& x) {
        return ToInt(IR::Value::Expr<int>(x.Load()));  // Already int
    }
    inline IR::Value::Expr<int> ToInt(const IR::Value::Var<float>& x) {
        return MakeCall<int>("int", Detail::BuildParams(IR::Value::Expr<float>(x.Load())));
    }
    inline IR::Value::Expr<int> ToInt(const IR::Value::Var<bool>& x) {
        return MakeCall<int>("int", Detail::BuildParams(IR::Value::Expr<bool>(x.Load())));
    }
    
    // ToInt from Expr
    inline IR::Value::Expr<int> ToInt(IR::Value::Expr<int>&& x) {
        return std::move(x);  // Already int, no conversion needed
    }
    inline IR::Value::Expr<int> ToInt(IR::Value::Expr<float>&& x) {
        return MakeCall<int>("int", Detail::BuildParams(std::move(x)));
    }
    inline IR::Value::Expr<int> ToInt(IR::Value::Expr<bool>&& x) {
        return MakeCall<int>("int", Detail::BuildParams(std::move(x)));
    }
    
    // ToBool from Var (explicit overloads to avoid ambiguity)
    inline IR::Value::Expr<bool> ToBool(const IR::Value::Var<bool>& x) {
        return ToBool(IR::Value::Expr<bool>(x.Load()));  // Already bool
    }
    inline IR::Value::Expr<bool> ToBool(const IR::Value::Var<float>& x) {
        return MakeCall<bool>("bool", Detail::BuildParams(IR::Value::Expr<float>(x.Load())));
    }
    inline IR::Value::Expr<bool> ToBool(const IR::Value::Var<int>& x) {
        return MakeCall<bool>("bool", Detail::BuildParams(IR::Value::Expr<int>(x.Load())));
    }
    
    // ToBool from Expr
    inline IR::Value::Expr<bool> ToBool(IR::Value::Expr<bool>&& x) {
        return std::move(x);  // Already bool, no conversion needed
    }
    inline IR::Value::Expr<bool> ToBool(IR::Value::Expr<float>&& x) {
        return MakeCall<bool>("bool", Detail::BuildParams(std::move(x)));
    }
    inline IR::Value::Expr<bool> ToBool(IR::Value::Expr<int>&& x) {
        return MakeCall<bool>("bool", Detail::BuildParams(std::move(x)));
    }
    
    // -------------------- Vector Type Casts (Float) --------------------
    
    // To Vec2
    inline IR::Value::Expr<Vec2> ToFloat(const IR::Value::Expr<IVec2>& x) {
        return MakeCall<Vec2>("vec2", Detail::BuildParams(x));
    }
    
    // To Vec3
    inline IR::Value::Expr<Vec3> ToFloat(const IR::Value::Expr<IVec3>& x) {
        return MakeCall<Vec3>("vec3", Detail::BuildParams(x));
    }
    
    // To Vec4
    inline IR::Value::Expr<Vec4> ToFloat(const IR::Value::Expr<IVec4>& x) {
        return MakeCall<Vec4>("vec4", Detail::BuildParams(x));
    }
    
    // -------------------- Vector Type Casts (Int) --------------------
    
    // To IVec2
    inline IR::Value::Expr<IVec2> ToInt(const IR::Value::Expr<Vec2>& x) {
        return MakeCall<IVec2>("ivec2", Detail::BuildParams(x));
    }
    
    // To IVec3
    inline IR::Value::Expr<IVec3> ToInt(const IR::Value::Expr<Vec3>& x) {
        return MakeCall<IVec3>("ivec3", Detail::BuildParams(x));
    }
    
    // To IVec4
    inline IR::Value::Expr<IVec4> ToInt(const IR::Value::Expr<Vec4>& x) {
        return MakeCall<IVec4>("ivec4", Detail::BuildParams(x));
    }
}

#endif //EASYGPU_MATH_H
