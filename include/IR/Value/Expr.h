#pragma once

/**
 * Expr.h:
 *      @Descripiton    :   The expression API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_EXPR_H
#define EASYGPU_EXPR_H

#include <IR/Value/Value.h>

#include <IR/Node/Operation.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/LoadUniform.h>

#include <IR/Builder/Builder.h>

#include <Utility/Vec.h>
#include <Utility/Matrix.h>
#include <Utility/Meta/StructMeta.h>

#include <sstream>
#include <memory>
#include <type_traits>
#include <concepts>
#include <string>

namespace GPU::IR::Value {
    // Forward declaration of Var for Expr constructor
    template<ScalarType T>
    class Var;
    
    // Forward declaration of ExprBase for CloneNode
    class ExprBase;
    
    // Helper to clone a node from const Expr - defined after ExprBase
    [[nodiscard]] std::unique_ptr<Node::Node> CloneNode(const ExprBase& expr);

    template<typename Type>
    std::string ValueToString(const Type &Value) {
        std::ostringstream oss;

        if constexpr (std::same_as<Type, float>) {
            oss << "float(" << Value << ")";
            return oss.str();
        } else if constexpr (std::same_as<Type, int>) {
            oss << "int(" << Value << ")";
            return oss.str();
        } else if constexpr (std::same_as<Type, bool>) {
            return Value ? "true" : "false";
        } else if constexpr (std::same_as<Type, Math::Vec2>) {
            oss << "vec2(float(" << Value.x << "), float(" << Value.y << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Vec3>) {
            oss << "vec3(float(" << Value.x << "), float(" << Value.y << "), float(" << Value.z << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Vec4>) {
            oss << "vec4(float(" << Value.x << "), float(" << Value.y << "), float(" << Value.z << "), float(" << Value.
                    w << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::IVec2>) {
            oss << "ivec2(int(" << Value.x << "), int(" << Value.y << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::IVec3>) {
            oss << "ivec3(int(" << Value.x << "), int(" << Value.y << "), int(" << Value.z << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::IVec4>) {
            oss << "ivec4(int(" << Value.x << "), int(" << Value.y << "), int(" << Value.z << "), int(" << Value.w << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat2>) {
            oss << "mat2(float(" << Value.m00 << "), float(" << Value.m10 << "), "
                    << "float(" << Value.m01 << "), float(" << Value.m11 << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat3>) {
            oss << "mat3("
                    << "float(" << Value.m00 << "), float(" << Value.m10 << "), float(" << Value.m20 << "), "
                    << "float(" << Value.m01 << "), float(" << Value.m11 << "), float(" << Value.m21 << "), "
                    << "float(" << Value.m02 << "), float(" << Value.m12 << "), float(" << Value.m22 << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat4>) {
            oss << "mat4("
                    << "float(" << Value.m00 << "), float(" << Value.m10 << "), float(" << Value.m20 << "), float(" <<
                    Value.m30 << "), "
                    << "float(" << Value.m01 << "), float(" << Value.m11 << "), float(" << Value.m21 << "), float(" <<
                    Value.m31 << "), "
                    << "float(" << Value.m02 << "), float(" << Value.m12 << "), float(" << Value.m22 << "), float(" <<
                    Value.m32 << "), "
                    << "float(" << Value.m03 << "), float(" << Value.m13 << "), float(" << Value.m23 << "), float(" <<
                    Value.m33 << "))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat2x3>) {
            oss << "mat2x3("
                    << "vec3(float(" << Value.c0.x << "), float(" << Value.c0.y << "), float(" << Value.c0.z << ")), "
                    << "vec3(float(" << Value.c1.x << "), float(" << Value.c1.y << "), float(" << Value.c1.z << ")))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat3x2>) {
            oss << "mat3x2("
                    << "vec2(float(" << Value.c0.x << "), float(" << Value.c0.y << ")), "
                    << "vec2(float(" << Value.c1.x << "), float(" << Value.c1.y << ")), "
                    << "vec2(float(" << Value.c2.x << "), float(" << Value.c2.y << ")))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat2x4>) {
            oss << "mat2x4("
                    << "vec4(float(" << Value.c0.x << "), float(" << Value.c0.y << "), float(" << Value.c0.z <<
                    "), float(" << Value.c0.w << ")), "
                    << "vec4(float(" << Value.c1.x << "), float(" << Value.c1.y << "), float(" << Value.c1.z <<
                    "), float(" << Value.c1.w << ")))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat4x2>) {
            oss << "mat4x2("
                    << "vec2(float(" << Value.c0.x << "), float(" << Value.c0.y << ")), "
                    << "vec2(float(" << Value.c1.x << "), float(" << Value.c1.y << ")), "
                    << "vec2(float(" << Value.c2.x << "), float(" << Value.c2.y << ")), "
                    << "vec2(float(" << Value.c3.x << "), float(" << Value.c3.y << ")))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat3x4>) {
            oss << "mat3x4("
                    << "vec4(float(" << Value.c0.x << "), float(" << Value.c0.y << "), float(" << Value.c0.z <<
                    "), float(" << Value.c0.w << ")), "
                    << "vec4(float(" << Value.c1.x << "), float(" << Value.c1.y << "), float(" << Value.c1.z <<
                    "), float(" << Value.c1.w << ")), "
                    << "vec4(float(" << Value.c2.x << "), float(" << Value.c2.y << "), float(" << Value.c2.z <<
                    "), float(" << Value.c2.w << ")))";
            return oss.str();
        } else if constexpr (std::same_as<Type, Math::Mat4x3>) {
            oss << "mat4x3("
                    << "vec3(float(" << Value.c0.x << "), float(" << Value.c0.y << "), float(" << Value.c0.z << ")), "
                    << "vec3(float(" << Value.c1.x << "), float(" << Value.c1.y << "), float(" << Value.c1.z << ")), "
                    << "vec3(float(" << Value.c2.x << "), float(" << Value.c2.y << "), float(" << Value.c2.z << ")), "
                    << "vec3(float(" << Value.c3.x << "), float(" << Value.c3.y << "), float(" << Value.c3.z << ")))";
            return oss.str();
        } else {
            return "unknown";
        }
    }

    /**
     * Base expression class (untyped/legacy support)
     * This is kept for backward compatibility and internal use
     */
    class ExprBase : public Value {
    public:
        ExprBase() = default;
        
        ExprBase(std::unique_ptr<Node::Node> Node) {
            _node = std::move(Node);
        }
        
        // Explicitly default move operations (base class Value has user-defined move, preventing auto-generation)
        ExprBase(ExprBase&&) = default;
        ExprBase& operator=(ExprBase&&) = default;
        
        // Copy operations are deleted (inherited from Value)
        ExprBase(const ExprBase&) = delete;
        ExprBase& operator=(const ExprBase&) = delete;

        ~ExprBase() = default;

    public:
        /**
         * Getting the node the expression owning
         * @return The node the expression owning
         */
        Node::Node *Node() const {
            return _node.get();
        }

    public:
        /**
         * If there is a statement or expression that really need unused, using this function
         * to ensure the IR building is correct
         */
        static void NotUse(ExprBase Expression) {
            Builder::Builder::Get().Build(*Expression.Node(), true);
        }
    };

    // Inline definition of CloneNode after ExprBase is defined
    [[nodiscard]] inline std::unique_ptr<Node::Node> CloneNode(const ExprBase& expr) {
        return expr.Node()->Clone();
    }

    // Forward declaration for friend declarations
    template<ScalarType T>
    class Expr;

    // Type traits to get element type of vector/matrix
    template<typename T>
    struct ElementType {
        using type = T;
    };
    
    template<> struct ElementType<Math::Vec2> { using type = float; };
    template<> struct ElementType<Math::Vec3> { using type = float; };
    template<> struct ElementType<Math::Vec4> { using type = float; };
    template<> struct ElementType<Math::IVec2> { using type = int; };
    template<> struct ElementType<Math::IVec3> { using type = int; };
    template<> struct ElementType<Math::IVec4> { using type = int; };
    
    // Vector dimension traits
    template<typename T>
    struct VecDimension {
        static constexpr int value = 0;
    };
    
    template<> struct VecDimension<Math::Vec2> { static constexpr int value = 2; };
    template<> struct VecDimension<Math::Vec3> { static constexpr int value = 3; };
    template<> struct VecDimension<Math::Vec4> { static constexpr int value = 4; };
    template<> struct VecDimension<Math::IVec2> { static constexpr int value = 2; };
    template<> struct VecDimension<Math::IVec3> { static constexpr int value = 3; };
    template<> struct VecDimension<Math::IVec4> { static constexpr int value = 4; };

    // Matrix traits
    template<typename T>
    struct MatrixTraits {
        static constexpr int cols = 0;
        static constexpr int rows = 0;
    };
    
    template<> struct MatrixTraits<Math::Mat2> { static constexpr int cols = 2; static constexpr int rows = 2; };
    template<> struct MatrixTraits<Math::Mat3> { static constexpr int cols = 3; static constexpr int rows = 3; };
    template<> struct MatrixTraits<Math::Mat4> { static constexpr int cols = 4; static constexpr int rows = 4; };
    template<> struct MatrixTraits<Math::Mat2x3> { static constexpr int cols = 2; static constexpr int rows = 3; };
    template<> struct MatrixTraits<Math::Mat2x4> { static constexpr int cols = 2; static constexpr int rows = 4; };
    template<> struct MatrixTraits<Math::Mat3x2> { static constexpr int cols = 3; static constexpr int rows = 2; };
    template<> struct MatrixTraits<Math::Mat3x4> { static constexpr int cols = 3; static constexpr int rows = 4; };
    template<> struct MatrixTraits<Math::Mat4x2> { static constexpr int cols = 4; static constexpr int rows = 2; };
    template<> struct MatrixTraits<Math::Mat4x3> { static constexpr int cols = 4; static constexpr int rows = 3; };

    /**
     * Typed expression class template
     * @tparam T The scalar type of the expression (float, int, bool, Vec2, Vec3, Vec4, Mat2, etc.)
     */
    template<ScalarType T>
    class Expr : public ExprBase {
    public:
        /**
         * The value type of this expression
         */
        using ValueType = T;
        
        /**
         * The element type (for vectors: float/int; for scalars: itself)
         */
        using ElementType_t = typename ElementType<T>::type;

    public:
        Expr() = default;
        
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        
        explicit Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        explicit Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        // Explicitly default move operations
        Expr(Expr&&) = default;
        Expr& operator=(Expr&&) = default;
        
        // Copy operations are deleted (inherited from Value via ExprBase)
        Expr(const Expr& expr) {
            _node = expr._node->Clone();
        }
        Expr& operator=(const Expr&) = delete;

        /**
         * Construct from same-type Var (explicit type check to prevent implicit conversion from wrong types)
         */
        Expr(const Var<T>& var);

        explicit Expr(const T& Value) {
            _node = std::make_unique<Node::LoadUniformNode>(ValueToString(Value));
        }

        /**
         * Construct from scalar value (for float/int/bool types)
         */
        template<typename U = T>
        Expr(T value)
            requires std::same_as<U, float> || std::same_as<U, int> || std::same_as<U, bool>{
            _node = std::make_unique<Node::LoadUniformNode>(ValueToString(value));
        }

        ~Expr() = default;

    public:
        /**
         * Subscript access for arrays
         */
        template<CountableType IndexType>
        Expr<ElementType_t> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<ElementType_t>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<BitableType IndexType>
        Expr<ElementType_t> operator[](IndexType index) & = delete;

        Expr<ElementType_t> operator[](ExprBase index) && {
            return Expr<ElementType_t>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }

        Expr<ElementType_t> operator[](ExprBase index) & = delete;
        
        // Allow typed expr as index
        template<ScalarType IndexT>
        Expr<ElementType_t> operator[](Expr<IndexT> index) && {
            return Expr<ElementType_t>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<ElementType_t> operator[](Expr<IndexT> index) & = delete;

    public:
        // Arithmetic operations: Expr<T> + Expr<T>
        friend Expr<T> operator+(const Expr<T>& lhs, const Expr<T>& rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator-(const Expr<T>& lhs, const Expr<T>& rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator*(const Expr<T>& lhs, const Expr<T>& rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator/(const Expr<T>& lhs, const Expr<T>& rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator%(const Expr<T>& lhs, const Expr<T>& rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mod, CloneNode(lhs), CloneNode(rhs)));
        }

        // Bitwise operations
        friend Expr<T> operator&(const Expr<T>& lhs, const Expr<T>& rhs) 
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator|(const Expr<T>& lhs, const Expr<T>& rhs)
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator^(const Expr<T>& lhs, const Expr<T>& rhs)
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator~(const Expr<T>& val)
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitNot, CloneNode(val)));
        }

        friend Expr<T> operator<<(const Expr<T>& lhs, const Expr<T>& rhs)
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shl, CloneNode(lhs), CloneNode(rhs)));
        }

        friend Expr<T> operator>>(const Expr<T>& lhs, const Expr<T>& rhs)
            requires BitableType<T> || std::same_as<T, Math::IVec2> || std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shr, CloneNode(lhs), CloneNode(rhs)));
        }

        // Bitwise operations with literal int (for int type)
        template<typename U = T>
        friend Expr<T> operator&(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator&(int lhs, Expr<T> rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitAnd,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<T> operator|(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator|(int lhs, Expr<T> rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitOr,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<T> operator^(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator^(int lhs, Expr<T> rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::BitXor,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<T> operator<<(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shl, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator>>(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Shr, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        // Modulo operation with literal int (for int type)
        template<typename U = T>
        friend Expr<T> operator%(Expr<T> lhs, int rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mod, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator%(int lhs, Expr<T> rhs)
            requires std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mod,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Comparison operations
        friend Expr<bool> operator<(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator==(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator!=(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator<=(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::LessEqual, lhs.Release(), rhs.Release()));
        }

        friend Expr<bool> operator>=(Expr<T> lhs, Expr<T> rhs) {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::GreaterEqual, lhs.Release(), rhs.Release()));
        }

        // Arithmetic with scalar: Expr<T> op scalar (only same element type)
        friend Expr<T> operator*(Expr<T> lhs, ElementType_t rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        friend Expr<T> operator*(ElementType_t lhs, Expr<T> rhs) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // For scalar types, also allow direct scalar operations
        template<typename U = T>
        friend Expr<T> operator+(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator+(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<T> operator-(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator-(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<T> operator/(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<T> operator/(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Comparison with scalar
        template<typename U = T>
        friend Expr<bool> operator<(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<bool> operator<(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Less,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<bool> operator>(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<bool> operator>(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Greater,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<bool> operator==(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int> || std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<bool> operator==(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int> || std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Equal,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        template<typename U = T>
        friend Expr<bool> operator!=(Expr<T> lhs, T rhs)
            requires std::same_as<U, float> || std::same_as<U, int> || std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual, lhs.Release(),
                    std::make_unique<Node::LoadUniformNode>(ValueToString(rhs))));
        }

        template<typename U = T>
        friend Expr<bool> operator!=(T lhs, Expr<T> rhs)
            requires std::same_as<U, float> || std::same_as<U, int> || std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::NotEqual,
                    std::make_unique<Node::LoadUniformNode>(ValueToString(lhs)), rhs.Release()));
        }

        // Logical operators - only for bool type
        template<typename U = T>
        friend Expr<bool> operator&&(const Expr<T>& lhs, const Expr<T>& rhs)
            requires std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::LogicalAnd, CloneNode(lhs), CloneNode(rhs)));
        }

        template<typename U = T>
        friend Expr<bool> operator||(const Expr<T>& lhs, const Expr<T>& rhs)
            requires std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::LogicalOr, CloneNode(lhs), CloneNode(rhs)));
        }

        template<typename U = T>
        friend Expr<bool> operator!(const Expr<T>& val)
            requires std::same_as<U, bool>
        {
            return Expr<bool>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::LogicalNot, CloneNode(val)));
        }

        // Unary minus for numeric types
        friend Expr<T> operator-(const Expr<T>& val) {
            return Expr<T>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Neg, CloneNode(val)));
        }
    };

    // Legacy alias for backward compatibility
    using ExprBase_t = ExprBase;

    // Common type aliases
    using BoolExpr = Expr<bool>;
    using FloatExpr = Expr<float>;
    using IntExpr = Expr<int>;
    using Vec2Expr = Expr<Math::Vec2>;
    using Vec3Expr = Expr<Math::Vec3>;
    using Vec4Expr = Expr<Math::Vec4>;
    using IVec2Expr = Expr<Math::IVec2>;
    using IVec3Expr = Expr<Math::IVec3>;
    using IVec4Expr = Expr<Math::IVec4>;
    using Mat2Expr = Expr<Math::Mat2>;
    using Mat3Expr = Expr<Math::Mat3>;
    using Mat4Expr = Expr<Math::Mat4>;
    using Mat2x3Expr = Expr<Math::Mat2x3>;
    using Mat2x4Expr = Expr<Math::Mat2x4>;
    using Mat3x2Expr = Expr<Math::Mat3x2>;
    using Mat3x4Expr = Expr<Math::Mat3x4>;
    using Mat4x2Expr = Expr<Math::Mat4x2>;
    using Mat4x3Expr = Expr<Math::Mat4x3>;
}

#endif //EASYGPU_EXPR_H
