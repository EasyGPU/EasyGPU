/**
 * Callable.h:
 *      @Descripiton    :   The callable function API for user-defined functions
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_CALLABLE_H
#define EASYGPU_CALLABLE_H

#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>
#include <IR/Node/Call.h>
#include <IR/Builder/Builder.h>
#include <IR/Builder/BuilderContext.h>
#include <Utility/Meta/StructMeta.h>

#include <string>
#include <functional>
#include <tuple>
#include <type_traits>
#include <memory>
#include <atomic>
#include <vector>

namespace GPU::Callables {
    namespace Detail {
        /**
         * Helper to remove reference from type
         * Var<int&> -> Var<int>
         */
        template<typename T>
        using RemoveRef = std::remove_reference_t<T>;

        /**
         * Helper to check if a type is a reference
         */
        template<typename T>
        inline constexpr bool IsRef = std::is_reference_v<T>;

        /**
         * Helper to get type name as string for GLSL
         */
        template<typename T>
        constexpr std::string_view GetGLSLTypeName() {
            using CleanT = RemoveRef<T>;
            if constexpr (std::same_as<CleanT, float>) return "float";
            else if constexpr (std::same_as<CleanT, int>) return "int";
            else if constexpr (std::same_as<CleanT, bool>) return "bool";
            else if constexpr (std::same_as<CleanT, Math::Vec2>) return "vec2";
            else if constexpr (std::same_as<CleanT, Math::Vec3>) return "vec3";
            else if constexpr (std::same_as<CleanT, Math::Vec4>) return "vec4";
            else if constexpr (std::same_as<CleanT, Math::IVec2>) return "ivec2";
            else if constexpr (std::same_as<CleanT, Math::IVec3>) return "ivec3";
            else if constexpr (std::same_as<CleanT, Math::IVec4>) return "ivec4";
            else if constexpr (std::same_as<CleanT, Math::Mat2>) return "mat2";
            else if constexpr (std::same_as<CleanT, Math::Mat3>) return "mat3";
            else if constexpr (std::same_as<CleanT, Math::Mat4>) return "mat4";
            else if constexpr (std::same_as<CleanT, Math::Mat2x3>) return "mat2x3";
            else if constexpr (std::same_as<CleanT, Math::Mat2x4>) return "mat2x4";
            else if constexpr (std::same_as<CleanT, Math::Mat3x2>) return "mat3x2";
            else if constexpr (std::same_as<CleanT, Math::Mat3x4>) return "mat3x4";
            else if constexpr (std::same_as<CleanT, Math::Mat4x2>) return "mat4x2";
            else if constexpr (std::same_as<CleanT, Math::Mat4x3>) return "mat4x3";
            // Support for registered structs
            else if constexpr (Meta::StructMeta<CleanT>::isRegistered) {
                return Meta::StructMeta<CleanT>::glslTypeName;
            }
            else return "unknown";
        }

        /**
         * Helper to generate unique function name
         */
        inline std::string GenerateUniqueFunctionName(const std::string &baseName) {
            static std::atomic<uint64_t> counter{0};
            uint64_t id = counter.fetch_add(1);
            if (baseName.empty()) {
                return "func_" + std::to_string(id);
            }
            return baseName + "_" + std::to_string(id);
        }

        /**
         * Helper to check if a type should be passed by reference (inout)
         * In GLSL, we use 'inout' qualifier for reference parameters
         */
        template<typename T>
        constexpr bool IsInOutType() {
            return std::is_reference_v<T>;
        }

        /**
         * Abstract base for type-erased callable body generator
         */
        class CallableBodyGeneratorBase {
        public:
            virtual ~CallableBodyGeneratorBase() = default;
            virtual void Generate() const = 0;
        };

        /**
         * Type-erased wrapper for the user's lambda
         * This avoids the need for std::function with reference parameters
         * 
         * Key insight: We use std::unique_ptr to store Var objects because
         * Var is non-copyable but unique_ptr is movable.
         */
        template<typename Func, typename... ParamTypes>
        class CallableBodyGenerator : public CallableBodyGeneratorBase {
        public:
            explicit CallableBodyGenerator(Func &&f) : _func(std::forward<Func>(f)) {}

            void Generate() const override {
                GenerateImpl(std::index_sequence_for<ParamTypes...>{});
            }

        private:
            template<size_t... Indices>
            void GenerateImpl(std::index_sequence<Indices...>) const {
                // Create unique_ptr to Var objects - Var is non-copyable
                // but unique_ptr is movable, and dereferencing gives us Var&
                auto vars = std::make_tuple(
                    std::make_unique<IR::Value::Var<ParamTypes>>(
                        "p" + std::to_string(Indices)
                    )...
                );
                
                // Call the function with dereferenced unique_ptrs (Var&)
                // This works for both Var<T> (by value/move) and Var<T>& (by reference) params
                _func(*std::get<Indices>(vars)...);
            }

            mutable Func _func;
        };
    }

    /**
     * Primary template - not defined, only used for function type syntax
     * Usage: Callable<float(float, float)> for a function returning float with two float args
     */
    template<typename Signature>
    class Callable;

    /**
     * Specialization for function types: R(Args...)
     * This allows syntax like Callable<float(float, float)>
     * @tparam R Return type
     * @tparam Args Argument types
     */
    template<typename R, typename... Args>
    class Callable<R(Args...)> {
    private:
        std::shared_ptr<Detail::CallableBodyGeneratorBase> _bodyGenerator;
        std::string _baseName;      // Base name for the function
        mutable std::string _mangledName;  // Generated unique name

    public:
        /**
         * Construct a callable with a definition function
         * @param def The function definition lambda that takes Var<Args>... and calls Return()
         * @param name Optional base name for the function
         */
        template<typename Func>
        Callable(Func &&def, std::string name = "") 
            : _baseName(std::move(name)) {
            // Type-erase the lambda using shared_ptr for easy copying
            _bodyGenerator = std::make_shared<Detail::CallableBodyGenerator<Func, Detail::RemoveRef<Args>...>>(
                std::forward<Func>(def)
            );
        }

        /**
         * Call the callable function, generating a CallInst expression
         * This will trigger function declaration/definition generation if not already done
         * @param args The argument expressions
         * @return An expression representing the function call
         */
        IR::Value::Expr<R> operator()(const IR::Value::Expr<Detail::RemoveRef<Args>> &... args) const {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) {
                // No active build context, return empty expression
                // This happens if called outside of Kernel construction
                return IR::Value::Expr<R>();
            }

            // Check if we've already generated this callable in this context
            // Use _bodyGenerator.get() as key to handle copied Callables correctly
            auto &state = context->GetCallableState(reinterpret_cast<const void*>(_bodyGenerator.get()));
            
            if (!state.declared) {
                // Generate unique function name
                _mangledName = Detail::GenerateUniqueFunctionName(_baseName);
                
                // Generate function prototype (forward declaration)
                std::string prototype = GeneratePrototype();
                context->AddCallableDeclaration(prototype);
                
                // Register function body generator (deferred until after main)
                context->AddCallableBodyGenerator([this]() {
                    GenerateFunctionBody();
                });
                
                state.declared = true;
            }

            // Create the call node
            std::vector<std::unique_ptr<IR::Node::Node>> argNodes;
            (argNodes.push_back(IR::Value::CloneNode(args)), ...);
            
            auto callNode = std::make_unique<IR::Node::CallNode>(_mangledName, std::move(argNodes));
            return IR::Value::Expr<R>(std::move(callNode));
        }

    private:
        /**
         * Generate the function prototype string
         */
        std::string GeneratePrototype() const {
            std::string result = std::string(Detail::GetGLSLTypeName<R>());
            result += " " + _mangledName + "(";
            
            // Generate parameter list
            std::vector<std::string> paramTypes = {std::string(Detail::GetGLSLTypeName<Args>())...};
            std::vector<bool> isInOut = {Detail::IsInOutType<Args>()...};
            for (size_t i = 0; i < paramTypes.size(); ++i) {
                if (i > 0) result += ", ";
                if (isInOut[i]) result += "inout ";
                result += paramTypes[i] + " p" + std::to_string(i);
            }
            
            result += ")";
            return result;
        }

        /**
         * Generate the function body by executing the user's definition lambda
         */
        void GenerateFunctionBody() const {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) return;

            // Mark that we're entering a callable body generation
            context->PushCallableBody();

            // Execute the type-erased body generator
            _bodyGenerator->Generate();
            
            // Pop the callable body context and get the generated code
            context->PopCallableBody();
        }
    };

    /**
     * Specialization for void return type: void(Args...)
     */
    template<typename... Args>
    class Callable<void(Args...)> {
    private:
        std::shared_ptr<Detail::CallableBodyGeneratorBase> _bodyGenerator;
        std::string _baseName;
        mutable std::string _mangledName;

    public:
        template<typename Func>
        Callable(Func &&def, std::string name = "")
            : _baseName(std::move(name)) {
            _bodyGenerator = std::make_shared<Detail::CallableBodyGenerator<Func, Detail::RemoveRef<Args>...>>(
                std::forward<Func>(def)
            );
        }

        void operator()(const IR::Value::Expr<Detail::RemoveRef<Args>> &... args) const {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) return;

            // Use _bodyGenerator.get() as key to handle copied Callables correctly
            auto &state = context->GetCallableState(reinterpret_cast<const void*>(_bodyGenerator.get()));
            
            if (!state.declared) {
                _mangledName = Detail::GenerateUniqueFunctionName(_baseName);
                
                std::string prototype = GeneratePrototype();
                context->AddCallableDeclaration(prototype);
                context->AddCallableBodyGenerator([this]() {
                    GenerateFunctionBody();
                });
                
                state.declared = true;
            }

            std::vector<std::unique_ptr<IR::Node::Node>> argNodes;
            (argNodes.push_back(IR::Value::CloneNode(args)), ...);
            
            auto callNode = std::make_unique<IR::Node::CallNode>(_mangledName, std::move(argNodes));
            IR::Builder::Builder::Get().Build(*callNode, true);
        }

    private:
        std::string GeneratePrototype() const {
            std::string result = "void " + _mangledName + "(";
            
            std::vector<std::string> paramTypes = {std::string(Detail::GetGLSLTypeName<Args>())...};
            std::vector<bool> isInOut = {Detail::IsInOutType<Args>()...};
            for (size_t i = 0; i < paramTypes.size(); ++i) {
                if (i > 0) result += ", ";
                if (isInOut[i]) result += "inout ";
                result += paramTypes[i] + " p" + std::to_string(i);
            }
            
            result += ")";
            return result;
        }

        void GenerateFunctionBody() const {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) return;

            context->PushCallableBody();

            _bodyGenerator->Generate();
            
            context->PopCallableBody();
        }
    };
}

#endif //EASYGPU_CALLABLE_H
