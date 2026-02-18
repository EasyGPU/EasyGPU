#pragma once

/**
 * VarBase.h:
 *      @Descripiton    :   The variable API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_VAR_H
#define EASYGPU_VAR_H

#include <IR/Value/Value.h>
#include <IR/Value/Expr.h>

#include <IR/Builder/Builder.h>

#include <IR/Node/LocalVariable.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LoadUniform.h>

#include <string>
#include <type_traits>
#include <concepts>
#include <memory>
#include <format>

namespace GPU::IR::Value {
    // Forward declarations for Var specializations (defined in separate headers)
    template<> class Var<Math::Vec2>;
    template<> class Var<Math::Vec3>;
    template<> class Var<Math::Vec4>;
    template<> class Var<Math::IVec2>;
    template<> class Var<Math::IVec3>;
    template<> class Var<Math::IVec4>;
    template<> class Var<Math::Mat2>;
    template<> class Var<Math::Mat3>;
    template<> class Var<Math::Mat4>;
    template<> class Var<Math::Mat2x3>;
    template<> class Var<Math::Mat2x4>;
    template<> class Var<Math::Mat3x2>;
    template<> class Var<Math::Mat3x4>;
    template<> class Var<Math::Mat4x2>;
    template<> class Var<Math::Mat4x3>;

    template<typename Type>
    constexpr const char *TypeShaderName() {
        if constexpr (std::same_as<Type, float>) return "float";
        else if constexpr (std::same_as<Type, int>) return "int";
        else if constexpr (std::same_as<Type, bool>) return "bool";
        else if constexpr (std::same_as<Type, Math::Vec2>) return "vec2";
        else if constexpr (std::same_as<Type, Math::Vec3>) return "vec3";
        else if constexpr (std::same_as<Type, Math::Vec4>) return "vec4";
        else if constexpr (std::same_as<Type, Math::IVec2>) return "ivec2";
        else if constexpr (std::same_as<Type, Math::IVec3>) return "ivec3";
        else if constexpr (std::same_as<Type, Math::IVec4>) return "ivec4";
        else if constexpr (std::same_as<Type, Math::Mat2>) return "mat2";
        else if constexpr (std::same_as<Type, Math::Mat3>) return "mat3";
        else if constexpr (std::same_as<Type, Math::Mat4>) return "mat4";
        else if constexpr (std::same_as<Type, Math::Mat2x3>) return "mat2x3";
        else if constexpr (std::same_as<Type, Math::Mat2x4>) return "mat2x4";
        else if constexpr (std::same_as<Type, Math::Mat3x2>) return "mat3x2";
        else if constexpr (std::same_as<Type, Math::Mat3x4>) return "mat3x4";
        else if constexpr (std::same_as<Type, Math::Mat4x2>) return "mat4x2";
        else if constexpr (std::same_as<Type, Math::Mat4x3>) return "mat4x3";
        else return "unknown";
    }

    /**
     * The variable API for users
     * @tparam Type The type of the var, must be scalar types
     */
    template<ScalarType Type>
    class VarBase : public Value {
    public:
        /**
         * Creating a variable through the VarBase API
         */
        VarBase() {
            auto name = Builder::Builder::Get().Context()->AssignVarName();

            _node    = std::make_unique<Node::LocalVariableNode>(name, TypeShaderName<Type>());
            _varNode = dynamic_cast<Node::LocalVariableNode *>(_node.get());

            // The variable definition is truly the statement
            Builder::Builder::Get().Build(*_varNode, true);
        }

        /**
         * Assignment from a expression
         * @param Value The expression to assign
         */
        explicit VarBase(Expr<Type> &&Value) noexcept {
            std::string name = Builder::Builder::Get().Context()->AssignVarName();

            _node    = std::make_unique<Node::LocalVariableNode>(name, TypeShaderName<Type>());
            _varNode = dynamic_cast<Node::LocalVariableNode *>(_node.get());

            auto lhs = Load();
            auto rhs = Value.Release();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));

            // The variable definition is truly the statement
            Builder::Builder::Get().Build(*_varNode, true);

            // Building the store node
            Builder::Builder::Get().Build(*store, true);
        }

        /**
        * Assignment from a expression
        * @param Value The expression to assign
        */
        explicit VarBase(Expr<Type> &Value) noexcept {
            std::string name = Builder::Builder::Get().Context()->AssignVarName();

            _node    = std::make_unique<Node::LocalVariableNode>(name, TypeShaderName<Type>());
            _varNode = dynamic_cast<Node::LocalVariableNode *>(_node.get());

            auto lhs = Load();
            auto rhs = Value.Release();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));

            // The variable definition is truly the statement
            Builder::Builder::Get().Build(*_varNode, true);

            // Building the store node
            Builder::Builder::Get().Build(*store, true);
        }

        /**
         * Creating a existed variable through the VarBase API, this API won't trigger the IR node tree construction
         * @param Name The name of the variable, if keep empty it will be assigned by the context
         */
        VarBase(std::string Name) {
            _node    = std::make_unique<Node::LocalVariableNode>(Name, TypeShaderName<Type>());
            _varNode = dynamic_cast<Node::LocalVariableNode *>(_node.get());
        }

        /**
         * Creating a reference to an external variable (e.g., uniform) through the VarBase API.
         * This API won't trigger the IR node tree construction and won't declare the variable in main().
         * @param Name The name of the external variable
         * @param IsExternal Flag to indicate this is an external variable (uniform, etc.)
         */
        VarBase(std::string Name, bool IsExternal) {
            _node    = std::make_unique<Node::LocalVariableNode>(Name, TypeShaderName<Type>(), IsExternal);
            _varNode = dynamic_cast<Node::LocalVariableNode *>(_node.get());
        }

        /**
         * Copy constructor - creates a new variable with value copied from source
         * @param Other The other VarBase to copy from
         */
        VarBase(const VarBase &Other)
            : VarBase() {
            // Copy value via IR load/store
            auto rhs = Other.Load();
            auto lhs = Load();
            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));
            Builder::Builder::Get().Build(*store, true);
        }

        /**
         * Copy constructor - creates a new variable with value copied from source
         * @param Other The other VarBase to copy from
         */
        VarBase(const VarBase&& Other)
            : VarBase() {
            // Copy value via IR load/store
            auto rhs = Other.Load();
            auto lhs = Load();
            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));
            Builder::Builder::Get().Build(*store, true);
        }

        /**
         * Move constructor
         * @param Other The other VarBase to move from
         */
        VarBase(VarBase &&Other) noexcept
            : Value(std::move(Other))
              , _varNode(Other._varNode) {
            Other._varNode = nullptr;
        }

        ~VarBase() = default;

        /**
         * Assignment from another VarBase produces IR load/store (not C++ copy)
         * @param Other The other to assign
         */
        VarBase &operator=(const VarBase &Other) {
            if (&Other == this) {
                return *this;
            }

            auto rhs = Other.Load();
            auto lhs = Load();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));
            Builder::Builder::Get().Build(*store, true);

            return *this;
        }

        /**
         * Assignment from another VarBase produces IR load/store (not C++ copy)
         * @param Other The other to assign
         */
        VarBase& operator=(const VarBase&& Other) {
            if (&Other == this) {
                return *this;
            }

            auto rhs = Other.Load();
            auto lhs = Load();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));
            Builder::Builder::Get().Build(*store, true);

            return *this;
        }

        /**
         * Assignment from a literal value produces a uniform load + store
         * @param Value The value to assign
         */
        VarBase &operator=(Type Value) {
            auto rhs = std::make_unique<Node::LoadUniformNode>(ValueToString<Type>(Value));
            auto lhs = Load();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));

            Builder::Builder::Get().Build(*store, true);

            return *this;
        }

        /**
         * Assignment from a expression
         * @param Value The expression to assign
         */
        VarBase &operator=(Expr<Type> Value) noexcept {
            auto lhs = Load();
            auto rhs = Value.Release();

            auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs));

            Builder::Builder::Get().Build(*store, true);

            return *this;
        }

    public:
        /**
         * Loading the variable to the IR node
         * @return The load node of this var
         */
        [[nodiscard]] std::unique_ptr<Node::LoadLocalVariableNode> Load() const {
            return std::make_unique<Node::LoadLocalVariableNode>(_varNode->VarName());
        }

    public:
        // VarBase op VarBase -> Expr<Type>
        template<ScalarType T>
        friend Expr<T> operator+(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator-(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator*(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator/(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator%(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator&(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator|(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator^(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;



        template<ScalarType T>
        friend Expr<T> operator<<(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator>>(const VarBase<T> &lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<bool> operator<(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator>(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator==(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator!=(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator<=(const VarBase<T> &lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator>=(const VarBase<T> &lhs, const VarBase<T> &rhs);

        // VarBase op Scalar -> Expr<T>
        template<ScalarType T>
        friend Expr<T> operator+(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<T> operator+(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator-(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<T> operator-(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator*(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<T> operator*(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator/(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<T> operator/(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<T> operator%(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator%(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator&(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator&(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator|(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator|(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator^(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator^(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator<<(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator<<(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator>>(const VarBase<T> &lhs, T rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<T> operator>>(T lhs, const VarBase<T> &rhs)
            requires BitableType<T>;

        template<ScalarType T>
        friend Expr<bool> operator<(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator<(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator>(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator>(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator==(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator==(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator!=(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator!=(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator<=(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator<=(T lhs, const VarBase<T> &rhs);

        template<ScalarType T>
        friend Expr<bool> operator>=(const VarBase<T> &lhs, T rhs);

        template<ScalarType T>
        friend Expr<bool> operator>=(T lhs, const VarBase<T> &rhs);

    public:
        VarBase &operator+=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::AddAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }
        VarBase &operator+=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::AddAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator-=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::SubAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator-=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::SubAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator*=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::MulAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator*=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::MulAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator/=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::DivAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator/=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::DivAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        // Compound assignment with Expr<Type>
        VarBase &operator+=(const Expr<Type> &other) {
            auto lhsLoad = this->Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::AddAssign,
                std::move(lhsLoad),
                CloneNode(other)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator-=(const Expr<Type> &other) {
            auto lhsLoad = this->Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::SubAssign,
                std::move(lhsLoad),
                CloneNode(other)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator*=(const Expr<Type> &other) {
            auto lhsLoad = this->Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::MulAssign,
                std::move(lhsLoad),
                CloneNode(other)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator/=(const Expr<Type> &other) {
            auto lhsLoad = this->Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::DivAssign,
                std::move(lhsLoad),
                CloneNode(other)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator%=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ModAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator%=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ModAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator&=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitAndAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator&=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitAndAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator|=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitOrAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator|=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitOrAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator^=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitXorAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }
        VarBase &operator^=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::BitXorAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator<<=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ShlAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }
        VarBase &operator<<=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ShlAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

        VarBase &operator>>=(const VarBase &other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = other.Load();
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ShrAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }
        VarBase &operator>>=(Type other) {
            auto lhsLoad = this->Load();
            auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(other));
            auto comAssign = std::make_unique<Node::CompoundAssignmentNode>(
                Node::CompoundAssignmentCode::ShrAssign,
                std::move(lhsLoad),
                std::move(rhsLoad)
            );

            Builder::Builder::Get().Build(*comAssign, true);

            return *this;
        }

    public:
        VarBase &operator++() {
            static_assert(std::is_same_v<Type, int>,
                "Prefix increment only supported for 'int'");

            auto increment = std::make_unique<Node::IncrementNode>(
                Node::IncrementDirection::Increment,
                this->Load(),
                true
            );
            Builder::Builder::Get().Build(*increment, true);

            return *this;
        }

        [[nodiscard("post-increment returns a new value; discarding it loses the result")]] Expr<Type> operator++(int) {
            static_assert(std::is_same_v<Type, int>,
                            "Postfix increment only supported for 'int'");

            auto increment = std::make_unique<Node::IncrementNode>(
                Node::IncrementDirection::Increment,
                this->Load(),
                false
            );

            return Expr<Type>(std::move(increment));
        }

        VarBase &operator--() {
            static_assert(std::is_same_v<Type, int>,
                "Prefix decrement only supported for 'int'");

            auto increment = std::make_unique<Node::IncrementNode>(
                Node::IncrementDirection::Decrement,
                this->Load(),
                true
            );
            Builder::Builder::Get().Build(*increment, true);

            return *this;
        }

        [[nodiscard("post-decrement returns a new value; discarding it loses the result")]] Expr<Type> operator--(int) {
            static_assert(std::is_same_v<Type, int>,
                            "Postfix decrement only supported for 'int'");

            auto increment = std::make_unique<Node::IncrementNode>(
                Node::IncrementDirection::Decrement,
                this->Load(),
                false
            );

            return Expr<Type>(std::move(increment));
        }

    public:
        /**
         * Convert to typed expression
         */
        operator Expr<Type>() {
            return Expr<Type>(std::move(Load()));
        }

    protected:
        Node::LocalVariableNode *_varNode;
        template<ScalarType U>
        friend class Var;
    };

    // Primary template for scalar types
    template<ScalarType Type>
    class Var : public VarBase<Type> {
    public:
        using VarBase<Type>::VarBase;
        using VarBase<Type>::Load;
        using VarBase<Type>::operator=;
    };

    // ==================== VarBase op VarBase ====================
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator+(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator*(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator/(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator%(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mod, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator&(const VarBase<Type> &lhs, const VarBase<Type> &rhs) 
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator|(const VarBase<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator^(const VarBase<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator~(const VarBase<Type> &var)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto load = var.Load();
        return Expr<Type>(std::make_unique<Node::OperationNode>(Node::OperationCode::BitNot, std::move(load)));
    }

    // Unary minus for numeric types
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(const VarBase<Type> &var)
        requires NumericType<Type>
    {
        auto load = var.Load();
        return Expr<Type>(std::make_unique<Node::OperationNode>(Node::OperationCode::Neg, std::move(load)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator<<(const VarBase<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator>>(const VarBase<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Less, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Greater, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator==(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Equal, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator!=(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<
                Node::OperationNode>(Node::OperationCode::NotEqual, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<=(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LessEqual, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>=(const VarBase<Type> &lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(std::make_unique<Node::OperationNode>(Node::OperationCode::GreaterEqual, std::move(lhsLoad),
                                                          std::move(rhsLoad)));
    }

    // ==================== VarBase op Scalar ====================
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator+(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator*(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator/(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator%(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mod, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator&(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator|(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator^(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator<<(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator>>(const VarBase<Type> &lhs, Type rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Less, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Greater, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator==(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Equal, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator!=(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<
                Node::OperationNode>(Node::OperationCode::NotEqual, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<=(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LessEqual, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>=(const VarBase<Type> &lhs, Type rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(std::make_unique<Node::OperationNode>(Node::OperationCode::GreaterEqual, std::move(lhsLoad),
                                                          std::move(rhsLoad)));
    }

    // ==================== Scalar op VarBase ====================
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator+(Type lhs, const VarBase<Type> &rhs) {
        return rhs + lhs;
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator*(Type lhs, const VarBase<Type> &rhs) {
        return rhs * lhs;
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator/(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator%(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mod, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator&(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator|(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator^(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator<<(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shl, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator>>(Type lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Shr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Less, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Greater, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator==(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Equal, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator!=(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<
                Node::OperationNode>(Node::OperationCode::NotEqual, std::move(lhsLoad), std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<=(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LessEqual, std::move(lhsLoad),
                                                  std::move(rhsLoad)));
    }

    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>=(Type lhs, const VarBase<Type> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(std::make_unique<Node::OperationNode>(Node::OperationCode::GreaterEqual, std::move(lhsLoad),
                                                          std::move(rhsLoad)));
    }

    // ============================================================================
    // Logical Operators for Var<bool> (&&, ||, !)
    // ============================================================================
    
    // Var<bool> && Var<bool>
    [[nodiscard]] inline Expr<bool> operator&&(const VarBase<bool> &lhs, const VarBase<bool> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    
    // Var<bool> || Var<bool>
    [[nodiscard]] inline Expr<bool> operator||(const VarBase<bool> &lhs, const VarBase<bool> &rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalOr, std::move(lhsLoad), std::move(rhsLoad)));
    }
    
    // !Var<bool>
    [[nodiscard]] inline Expr<bool> operator!(const VarBase<bool> &var) {
        auto load = var.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalNot, std::move(load)));
    }
    
    // Var<bool> && bool literal
    [[nodiscard]] inline Expr<bool> operator&&(const VarBase<bool> &lhs, bool rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    
    // bool literal && Var<bool>
    [[nodiscard]] inline Expr<bool> operator&&(bool lhs, const VarBase<bool> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalAnd, std::move(lhsLoad), std::move(rhsLoad)));
    }
    
    // Var<bool> || bool literal
    [[nodiscard]] inline Expr<bool> operator||(const VarBase<bool> &lhs, bool rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalOr, std::move(lhsLoad), std::move(rhsLoad)));
    }
    
    // bool literal || Var<bool>
    [[nodiscard]] inline Expr<bool> operator||(bool lhs, const VarBase<bool> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalOr, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Var<bool> && Expr<bool>
    [[nodiscard]] inline Expr<bool> operator&&(const VarBase<bool> &lhs, const Expr<bool> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalAnd, lhs.Load(), CloneNode(rhs)));
    }
    
    // Expr<bool> && Var<bool>
    [[nodiscard]] inline Expr<bool> operator&&(const Expr<bool> &lhs, const VarBase<bool> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalAnd, CloneNode(lhs), rhs.Load()));
    }
    
    // Var<bool> || Expr<bool>
    [[nodiscard]] inline Expr<bool> operator||(const VarBase<bool> &lhs, const Expr<bool> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalOr, lhs.Load(), CloneNode(rhs)));
    }
    
    // Expr<bool> || Var<bool>
    [[nodiscard]] inline Expr<bool> operator||(const Expr<bool> &lhs, const VarBase<bool> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LogicalOr, CloneNode(lhs), rhs.Load()));
    }

    // ============================================================================
    // Var-Expr and Expr-Var Cross Operators
    // ============================================================================
    
    // Arithmetic: VarBase<T> op Expr<T>
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator+(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator*(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator/(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator%(const VarBase<Type> &lhs, const Expr<Type> &rhs)
        requires BitableType<Type>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mod, lhs.Load(), CloneNode(rhs)));
    }
    
    // Arithmetic: Expr<T> op VarBase<T>
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator+(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Add, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator-(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Sub, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator*(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator/(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator%(const Expr<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mod, CloneNode(lhs), rhs.Load()));
    }
    
    // Comparison: VarBase<T> op Expr<T>
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Less, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Greater, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator==(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Equal, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator!=(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::NotEqual, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<=(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LessEqual, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>=(const VarBase<Type> &lhs, const Expr<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::GreaterEqual, lhs.Load(), CloneNode(rhs)));
    }
    
    // Comparison: Expr<T> op VarBase<T>
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Less, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Greater, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator==(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Equal, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator!=(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::NotEqual, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator<=(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::LessEqual, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<bool> operator>=(const Expr<Type> &lhs, const VarBase<Type> &rhs) {
        return Expr<bool>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::GreaterEqual, CloneNode(lhs), rhs.Load()));
    }
    
    // Bitwise: VarBase<T> op Expr<T> (for bitable types)
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator&(const VarBase<Type> &lhs, const Expr<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator|(const VarBase<Type> &lhs, const Expr<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, lhs.Load(), CloneNode(rhs)));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator^(const VarBase<Type> &lhs, const Expr<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, lhs.Load(), CloneNode(rhs)));
    }
    
    // Bitwise: Expr<T> op VarBase<T>
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator&(const Expr<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitAnd, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator|(const Expr<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitOr, CloneNode(lhs), rhs.Load()));
    }
    
    template<ScalarType Type>
    [[nodiscard]] Expr<Type> operator^(const Expr<Type> &lhs, const VarBase<Type> &rhs)
        requires BitableType<Type> || std::same_as<Type, Math::IVec2> || std::same_as<Type, Math::IVec3> || std::same_as<Type, Math::IVec4>
    {
        return Expr<Type>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::BitXor, CloneNode(lhs), rhs.Load()));
    }

    // ============================================================================
    // Expr Constructors from Var (must be defined after VarBase)
    // ============================================================================
    
    /**
     * Construct Expr from same-type Var (enforces type safety)
     */
    template<ScalarType T>
    Expr<T>::Expr(const Var<T>& var) : ExprBase(var.Load()) {}
}

// Include Var specializations for vector and matrix types (after main template definition)
#include <IR/Value/VarVector.h>
#include <IR/Value/VarIVector.h>
#include <IR/Value/VarMatrix.h>

// Expr constructor implementations for vector types (after specializations)
namespace GPU::IR::Value {
    inline Expr<Math::Vec2>::Expr(const Var<Math::Vec2>& var) : ExprBase(var.Load()) {}
    inline Expr<Math::Vec3>::Expr(const Var<Math::Vec3>& var) : ExprBase(var.Load()) {}
    inline Expr<Math::Vec4>::Expr(const Var<Math::Vec4>& var) : ExprBase(var.Load()) {}
    inline Expr<Math::IVec2>::Expr(const Var<Math::IVec2>& var) : ExprBase(var.Load()) {}
    inline Expr<Math::IVec3>::Expr(const Var<Math::IVec3>& var) : ExprBase(var.Load()) {}
    inline Expr<Math::IVec4>::Expr(const Var<Math::IVec4>& var) : ExprBase(var.Load()) {}
}

#endif //EASYGPU_VAR_H
