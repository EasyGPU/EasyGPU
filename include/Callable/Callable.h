#pragma once

/**
 * Callable.h:
 *      @Descripiton    :   The callable function API for user-defined functions
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_CALLABLE_H
#define EASYGPU_CALLABLE_H

#include <IR/Builder/Builder.h>
#include <IR/Builder/BuilderContext.h>
#include <IR/Node/Call.h>
#include <IR/Value/Expr.h>
#include <IR/Value/SideEffectToken.h>
#include <IR/Value/Var.h>
#include <Utility/Meta/StructMeta.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace GPU::Callables {
namespace Detail {
/**
 * Helper to remove reference from type
 * Var<int&> -> Var<int>
 */
template <typename T> using RemoveRef			  = std::remove_reference_t<T>;

/**
 * Helper to check if a type is a reference
 */
template <typename T> inline constexpr bool IsRef = std::is_reference_v<T>;

// Forward declarations for type extraction
namespace TypeExtractor {
// Primary template: assume T is already a scalar type
template <typename T> struct ExtractScalar {
	using type						= T;
	static constexpr bool isGpuType = false;
};

// Specialization for Var<T>
template <typename T> struct ExtractScalar<IR::Value::Var<T>> {
	using type						= T;
	static constexpr bool isGpuType = true;
};

// Specialization for Expr<T>
template <typename T> struct ExtractScalar<IR::Value::Expr<T>> {
	using type						= T;
	static constexpr bool isGpuType = true;
};

// Helper alias
template <typename T> using ToScalar				  = typename ExtractScalar<RemoveRef<T>>::type;

template <typename T> inline constexpr bool IsGpuType = ExtractScalar<RemoveRef<T>>::isGpuType;
} // namespace TypeExtractor

/**
 * Helper to get type name as string for GLSL
 * Supports both C++ types (float, Math::Vec3) and GPU types (Float, Float3)
 */
template <typename T> constexpr std::string_view GetGLSLTypeName() {
	using CleanT = TypeExtractor::ToScalar<T>;
	if constexpr (std::same_as<CleanT, float>)
		return "float";
	else if constexpr (std::same_as<CleanT, int>)
		return "int";
	else if constexpr (std::same_as<CleanT, bool>)
		return "bool";
	else if constexpr (std::same_as<CleanT, Math::Vec2>)
		return "vec2";
	else if constexpr (std::same_as<CleanT, Math::Vec3>)
		return "vec3";
	else if constexpr (std::same_as<CleanT, Math::Vec4>)
		return "vec4";
	else if constexpr (std::same_as<CleanT, Math::IVec2>)
		return "ivec2";
	else if constexpr (std::same_as<CleanT, Math::IVec3>)
		return "ivec3";
	else if constexpr (std::same_as<CleanT, Math::IVec4>)
		return "ivec4";
	else if constexpr (std::same_as<CleanT, Math::Mat2>)
		return "mat2";
	else if constexpr (std::same_as<CleanT, Math::Mat3>)
		return "mat3";
	else if constexpr (std::same_as<CleanT, Math::Mat4>)
		return "mat4";
	else if constexpr (std::same_as<CleanT, Math::Mat2x3>)
		return "mat2x3";
	else if constexpr (std::same_as<CleanT, Math::Mat2x4>)
		return "mat2x4";
	else if constexpr (std::same_as<CleanT, Math::Mat3x2>)
		return "mat3x2";
	else if constexpr (std::same_as<CleanT, Math::Mat3x4>)
		return "mat3x4";
	else if constexpr (std::same_as<CleanT, Math::Mat4x2>)
		return "mat4x2";
	else if constexpr (std::same_as<CleanT, Math::Mat4x3>)
		return "mat4x3";
	// Support for registered structs
	else if constexpr (Meta::StructMeta<CleanT>::isRegistered) {
		return Meta::StructMeta<CleanT>::glslTypeName;
	} else
		return "unknown";
}

/**
 * Helper to generate unique function name
 */
inline std::string GenerateUniqueFunctionName(const std::string &baseName) {
	static std::atomic<uint64_t> counter{0};
	uint64_t					 id = counter.fetch_add(1);
	if (baseName.empty()) {
		return "func_" + std::to_string(id);
	}
	return baseName + "_" + std::to_string(id);
}

/**
 * Helper to check if a type should be passed by reference (inout)
 * In GLSL, we use 'inout' qualifier for reference parameters
 */
template <typename T> constexpr bool IsInOutType() {
	return std::is_reference_v<T>;
}

/**
 * Abstract base for type-erased callable body generator
 */
class CallableBodyGeneratorBase {
public:
	virtual ~CallableBodyGeneratorBase() = default;
	virtual void Generate() const		 = 0;
};

/**
 * Type-erased wrapper for the user's lambda
 * This avoids the need for std::function with reference parameters
 *
 * Key insight: We use std::unique_ptr to store Var objects because
 * Var is non-copyable but unique_ptr is movable.
 */
template <typename Func, typename... ParamTypes> class CallableBodyGenerator : public CallableBodyGeneratorBase {
public:
	explicit CallableBodyGenerator(Func &&f) : _func(std::forward<Func>(f)) {
	}

	void Generate() const override {
		GenerateImpl(std::index_sequence_for<ParamTypes...>{});
	}

private:
	template <size_t... Indices> void GenerateImpl(std::index_sequence<Indices...>) const {
		// Create Var objects that reference the GLSL parameter directly
		// isExternal=true means no local variable declaration is generated
		auto vars = std::make_tuple(IR::Value::Var<ParamTypes>("p" + std::to_string(Indices),
															   true // isExternal - use parameter name directly, no copy
															   )...);

		// Call the function with Var& that reference parameters directly
		_func(std::get<Indices>(vars)...);
	}

	mutable Func _func;
};
} // namespace Detail

/**
 * Primary template - not defined, only used for function type syntax
 * Usage: Callable<float(float, float)> for a function returning float with two float args
 */
template <typename Signature> class Callable;

/**
 * Specialization for function types: R(Args...)
 * This allows syntax like Callable<float(float, float)>
 * @tparam R Return type
 * @tparam Args Argument types
 */
template <typename R, typename... Args> class Callable<R(Args...)> {
private:
	std::shared_ptr<Detail::CallableBodyGeneratorBase> _bodyGenerator;
	std::string										   _baseName;	 // Base name for the function
	mutable std::string								   _mangledName; // Generated unique name

	// Extract scalar types for Args and R (supports both C++ types and GPU types like Float, Float3)
	using ScalarR							= Detail::TypeExtractor::ToScalar<R>;
	template <typename Arg> using ScalarArg = Detail::TypeExtractor::ToScalar<Arg>;

public:
	/**
	 * Construct a callable with a definition function
	 * @param def The function definition lambda that takes Var<Args>... and calls Return()
	 * @param name Optional base name for the function
	 */
	template <typename Func> Callable(Func &&def, std::string name = "") : _baseName(std::move(name)) {
		// Type-erase the lambda using shared_ptr for easy copying
		// Use scalar types for the body generator (Var<float> needs float)
		_bodyGenerator =
			std::make_shared<Detail::CallableBodyGenerator<Func, ScalarArg<Args>...>>(std::forward<Func>(def));
	}

	/**
	 * Call the callable function, generating a CallInst expression
	 * This will trigger function declaration/definition generation if not already done
	 * @param args The argument expressions (can be Expr<T> or Var<T>)
	 * @return An expression representing the function call
	 */
	IR::Value::Expr<ScalarR> operator()(const IR::Value::Expr<ScalarArg<Args>> &...args) const {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			// No active build context, return empty expression
			// This happens if called outside of Kernel construction
			return IR::Value::Expr<ScalarR>();
		}

		// Check if we've already generated this callable in this context
		// Use _bodyGenerator.get() as key to handle copied Callables correctly
		auto &state = context->GetCallableState(reinterpret_cast<const void *>(_bodyGenerator.get()));

		if (!state.declared) {
			// Generate unique function name
			_mangledName		  = Detail::GenerateUniqueFunctionName(_baseName);

			// Generate function prototype (forward declaration)
			std::string prototype = GeneratePrototype();
			context->AddCallableDeclaration(prototype);

			// Register function body generator (deferred until after main)
			context->AddCallableBodyGenerator([this]() { GenerateFunctionBody(); });

			state.declared = true;
		}

		// Create the call node
		std::vector<std::unique_ptr<IR::Node::Node>> argNodes;
		(argNodes.push_back(IR::Value::CloneNode(args)), ...);

		auto callNode = std::make_unique<IR::Node::CallNode>(_mangledName, std::move(argNodes));
		return IR::Value::Expr<ScalarR>(std::move(callNode));
	}

private:
	/**
	 * Generate the function prototype string
	 */
	std::string GeneratePrototype() const {
		std::string result					 = std::string(Detail::GetGLSLTypeName<R>());
		result								+= " " + _mangledName + "(";

		// Generate parameter list (use scalar types for type names, but original Args for inout check)
		std::vector<std::string> paramTypes	 = {std::string(Detail::GetGLSLTypeName<ScalarArg<Args>>())...};
		std::vector<bool>		 isInOut	 = {Detail::IsInOutType<Args>()...};
		for (size_t i = 0; i < paramTypes.size(); ++i) {
			if (i > 0)
				result += ", ";
			if (isInOut[i])
				result += "inout ";
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
		if (!context)
			return;

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
template <typename... Args> class Callable<void(Args...)> {
private:
	std::shared_ptr<Detail::CallableBodyGeneratorBase> _bodyGenerator;
	std::string										   _baseName;
	mutable std::string								   _mangledName;

	// Extract scalar types for Args (supports both C++ types and GPU types)
	template <typename Arg> using ScalarArg = Detail::TypeExtractor::ToScalar<Arg>;

public:
	template <typename Func> Callable(Func &&def, std::string name = "") : _baseName(std::move(name)) {
		// Use scalar types for the body generator
		_bodyGenerator =
			std::make_shared<Detail::CallableBodyGenerator<Func, ScalarArg<Args>...>>(std::forward<Func>(def));
	}

	[[nodiscard]]
	IR::Value::SideEffectToken operator()(const IR::Value::Expr<ScalarArg<Args>> &...args) const {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			// No active build context, return empty token
			return IR::Value::SideEffectToken(nullptr);
		}

		// Use _bodyGenerator.get() as key to handle copied Callables correctly
		auto &state = context->GetCallableState(reinterpret_cast<const void *>(_bodyGenerator.get()));

		if (!state.declared) {
			_mangledName		  = Detail::GenerateUniqueFunctionName(_baseName);

			std::string prototype = GeneratePrototype();
			context->AddCallableDeclaration(prototype);
			context->AddCallableBodyGenerator([this]() { GenerateFunctionBody(); });

			state.declared = true;
		}

		std::vector<std::unique_ptr<IR::Node::Node>> argNodes;
		(argNodes.push_back(IR::Value::CloneNode(args)), ...);

		auto callNode = std::make_unique<IR::Node::CallNode>(_mangledName, std::move(argNodes));

		// Return a token that will commit the side effect on destruction
		return IR::Value::SideEffectToken(std::move(callNode));
	}

private:
	std::string GeneratePrototype() const {
		std::string				 result		= "void " + _mangledName + "(";

		std::vector<std::string> paramTypes = {std::string(Detail::GetGLSLTypeName<ScalarArg<Args>>())...};
		// Use original Args (not ScalarArg) for inout check, as ScalarArg removes reference
		std::vector<bool>		 isInOut	= {Detail::IsInOutType<Args>()...};
		for (size_t i = 0; i < paramTypes.size(); ++i) {
			if (i > 0)
				result += ", ";
			if (isInOut[i])
				result += "inout ";
			result += paramTypes[i] + " p" + std::to_string(i);
		}

		result += ")";
		return result;
	}

	void GenerateFunctionBody() const {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context)
			return;

		context->PushCallableBody();

		_bodyGenerator->Generate();

		context->PopCallableBody();
	}
};
} // namespace GPU::Callables

#endif // EASYGPU_CALLABLE_H
