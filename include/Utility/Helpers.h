#pragma once

/**
 * Helpers.h:
 *      @Descripiton    :   Helper functions and type aliases for EasyGPU
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 *      @Modified       :   MakeFloat3 broadcast optimization
 */
#ifndef EASYGPU_HELPERS_H
#define EASYGPU_HELPERS_H

// X11 defines Bool as typedef int Bool, which conflicts with our Bool alias
#ifdef Bool
#undef Bool
#endif

#include <IR/Builder/Builder.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Value/Var.h>
#include <Utility/Math.h>

#include <format>

namespace GPU {
// ============================================================================
// Type Aliases for Var types
// ============================================================================
namespace Alias {
using Int		 = IR::Value::Var<int>;
using Float		 = IR::Value::Var<float>;
using Bool		 = IR::Value::Var<bool>;
using Float2	 = IR::Value::Var<Math::Vec2>;
using Float3	 = IR::Value::Var<Math::Vec3>;
using Float4	 = IR::Value::Var<Math::Vec4>;
using Int2		 = IR::Value::Var<Math::IVec2>;
using Int3		 = IR::Value::Var<Math::IVec3>;
using Int4		 = IR::Value::Var<Math::IVec4>;
using Matrix2	 = IR::Value::Var<Math::Mat2>;
using Matrix3	 = IR::Value::Var<Math::Mat3>;
using Matrix4	 = IR::Value::Var<Math::Mat4>;
using Matrix2x3	 = IR::Value::Var<Math::Mat2x3>;
using Matrix3x2	 = IR::Value::Var<Math::Mat3x2>;
using Matrix2x4	 = IR::Value::Var<Math::Mat2x4>;
using Matrix4x2	 = IR::Value::Var<Math::Mat4x2>;
using Matrix3x4	 = IR::Value::Var<Math::Mat3x4>;
using Matrix4x3	 = IR::Value::Var<Math::Mat4x3>;
using IntExpr	 = IR::Value::Expr<int>;
using FloatExpr	 = IR::Value::Expr<float>;
using BoolExpr	 = IR::Value::Expr<bool>;
using Float2Expr = IR::Value::Expr<Math::Vec2>;
using Float3Expr = IR::Value::Expr<Math::Vec3>;
using Float4Expr = IR::Value::Expr<Math::Vec4>;
using Int2Expr	 = IR::Value::Expr<Math::IVec2>;
using Int3Expr	 = IR::Value::Expr<Math::IVec3>;
using Int4Expr	 = IR::Value::Expr<Math::IVec4>;
using Mat2Expr	 = IR::Value::Expr<Math::Mat2>;
using Mat3Expr	 = IR::Value::Expr<Math::Mat3>;
using Mat4Expr	 = IR::Value::Expr<Math::Mat4>;
using Mat2x3Expr = IR::Value::Expr<Math::Mat2x3>;
using Mat3x2Expr = IR::Value::Expr<Math::Mat3x2>;
using Mat2x4Expr = IR::Value::Expr<Math::Mat2x4>;
using Mat4x2Expr = IR::Value::Expr<Math::Mat4x2>;
using Mat3x4Expr = IR::Value::Expr<Math::Mat3x4>;
using Mat4x3Expr = IR::Value::Expr<Math::Mat4x3>;
} // namespace Alias

// ============================================================================
// Helper functions to construct vectors from components
// ============================================================================
namespace Construct {
namespace Detail {
// Trait to detect if T is Var<U>
template <typename T, typename U> struct IsVarOf : std::false_type {};
template <typename U> struct IsVarOf<IR::Value::Var<U>, U> : std::true_type {};

// Trait to detect if T is Expr<U>
template <typename T, typename U> struct IsExprOf : std::false_type {};
template <typename U> struct IsExprOf<IR::Value::Expr<U>, U> : std::true_type {};

// Concept for valid float component type
template <typename T>
concept FloatComponent =
	std::same_as<std::remove_cvref_t<T>, float> || std::same_as<std::remove_cvref_t<T>, double> ||
	IsVarOf<std::remove_cvref_t<T>, float>::value || IsExprOf<std::remove_cvref_t<T>, float>::value;

// Concept for valid int component type
template <typename T>
concept IntComponent = std::convertible_to<std::remove_cvref_t<T>, int> ||
					   IsVarOf<std::remove_cvref_t<T>, int>::value || IsExprOf<std::remove_cvref_t<T>, int>::value;

// Concept for valid Vec2/3/4 component type
template <typename T>
concept Vec2Component =
	std::same_as<std::remove_cvref_t<T>, Math::Vec2> || IsVarOf<std::remove_cvref_t<T>, Math::Vec2>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::Vec2>::value;

template <typename T>
concept Vec3Component =
	std::same_as<std::remove_cvref_t<T>, Math::Vec3> || IsVarOf<std::remove_cvref_t<T>, Math::Vec3>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::Vec3>::value;

template <typename T>
concept Vec4Component =
	std::same_as<std::remove_cvref_t<T>, Math::Vec4> || IsVarOf<std::remove_cvref_t<T>, Math::Vec4>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::Vec4>::value;

// Integer vector component concepts
template <typename T>
concept IVec2Component =
	std::same_as<std::remove_cvref_t<T>, Math::IVec2> || IsVarOf<std::remove_cvref_t<T>, Math::IVec2>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::IVec2>::value;

template <typename T>
concept IVec3Component =
	std::same_as<std::remove_cvref_t<T>, Math::IVec3> || IsVarOf<std::remove_cvref_t<T>, Math::IVec3>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::IVec3>::value;

template <typename T>
concept IVec4Component =
	std::same_as<std::remove_cvref_t<T>, Math::IVec4> || IsVarOf<std::remove_cvref_t<T>, Math::IVec4>::value ||
	IsExprOf<std::remove_cvref_t<T>, Math::IVec4>::value;

// Helper to convert value to GLSL string representation
template <typename T> [[nodiscard]] inline std::string ToGLSLString(T &&val) {
	using U = std::remove_cvref_t<T>;
	if constexpr (std::same_as<U, float> || std::same_as<U, double> || std::same_as<U, int>) {
		return std::to_string(static_cast<float>(val));
	} else if constexpr (IsExprOf<U, float>::value) {
		return IR::Builder::Builder::Get().BuildNode(*val.Node());
	} else if constexpr (IsVarOf<U, float>::value) {
		return IR::Builder::Builder::Get().BuildNode(*val.Load().get());
	} else if constexpr (IsExprOf<U, Math::Vec2>::value || IsExprOf<U, Math::Vec3>::value ||
						 IsExprOf<U, Math::Vec4>::value) {
		return IR::Builder::Builder::Get().BuildNode(*val.Node());
	} else if constexpr (IsVarOf<U, Math::Vec2>::value || IsVarOf<U, Math::Vec3>::value ||
						 IsVarOf<U, Math::Vec4>::value) {
		return IR::Builder::Builder::Get().BuildNode(*val.Load().get());
	} else {
		return "";
	}
}

// Helper to build parameter list for vector constructors (for multi-component)
inline std::vector<std::unique_ptr<IR::Node::Node>> BuildVectorParams(const std::vector<IR::Value::ExprBase *> &exprs) {
	std::vector<std::unique_ptr<IR::Node::Node>> params;
	params.reserve(exprs.size());
	for (auto *expr : exprs) {
		params.push_back(const_cast<IR::Value::ExprBase *>(expr)->Node()->Clone());
	}
	return params;
}

// Internal implementations that take Expr (for multi-component)
[[nodiscard]] inline IR::Value::Expr<Math::Vec2> MakeFloat2Impl(const IR::Value::Expr<float> &x,
																const IR::Value::Expr<float> &y) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<float> *>(&x),
												const_cast<IR::Value::Expr<float> *>(&y)};
	return Math::MakeCall<Math::Vec2>("vec2", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::Vec3> MakeFloat3Impl(const IR::Value::Expr<float> &x,
																const IR::Value::Expr<float> &y,
																const IR::Value::Expr<float> &z) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<float> *>(&x),
												const_cast<IR::Value::Expr<float> *>(&y),
												const_cast<IR::Value::Expr<float> *>(&z)};
	return Math::MakeCall<Math::Vec3>("vec3", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4Impl(const IR::Value::Expr<float> &x,
																const IR::Value::Expr<float> &y,
																const IR::Value::Expr<float> &z,
																const IR::Value::Expr<float> &w) {
	std::vector<IR::Value::ExprBase *> exprs = {
		const_cast<IR::Value::Expr<float> *>(&x), const_cast<IR::Value::Expr<float> *>(&y),
		const_cast<IR::Value::Expr<float> *>(&z), const_cast<IR::Value::Expr<float> *>(&w)};
	return Math::MakeCall<Math::Vec4>("vec4", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec2> MakeInt2Impl(const IR::Value::Expr<int> &x,
															   const IR::Value::Expr<int> &y) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<int> *>(&x),
												const_cast<IR::Value::Expr<int> *>(&y)};
	return Math::MakeCall<Math::IVec2>("ivec2", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec3> MakeInt3Impl(const IR::Value::Expr<int> &x,
															   const IR::Value::Expr<int> &y,
															   const IR::Value::Expr<int> &z) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<int> *>(&x),
												const_cast<IR::Value::Expr<int> *>(&y),
												const_cast<IR::Value::Expr<int> *>(&z)};
	return Math::MakeCall<Math::IVec3>("ivec3", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4Impl(const IR::Value::Expr<int> &x,
															   const IR::Value::Expr<int> &y,
															   const IR::Value::Expr<int> &z,
															   const IR::Value::Expr<int> &w) {
	std::vector<IR::Value::ExprBase *> exprs = {
		const_cast<IR::Value::Expr<int> *>(&x), const_cast<IR::Value::Expr<int> *>(&y),
		const_cast<IR::Value::Expr<int> *>(&z), const_cast<IR::Value::Expr<int> *>(&w)};
	return Math::MakeCall<Math::IVec4>("ivec4", BuildVectorParams(exprs));
}

// Trait to detect Vec types (for disabling lazy fill when Vec is passed)
template <typename T> struct IsVecType : std::false_type {};
template <> struct IsVecType<Math::Vec2> : std::true_type {};
template <> struct IsVecType<Math::Vec3> : std::true_type {};
template <> struct IsVecType<Math::Vec4> : std::true_type {};
template <> struct IsVecType<Math::IVec2> : std::true_type {};
template <> struct IsVecType<Math::IVec3> : std::true_type {};
template <> struct IsVecType<Math::IVec4> : std::true_type {};

// Trait to detect if T is a scalar type (not Var/Expr)
template <typename T> struct IsScalarType : std::false_type {};
template <> struct IsScalarType<float> : std::true_type {};
template <> struct IsScalarType<double> : std::true_type {};
template <> struct IsScalarType<int> : std::true_type {};

// Helper to convert scalar to float
[[nodiscard]] inline float ScalarToFloat(double val) {
	return static_cast<float>(val);
}
[[nodiscard]] inline float ScalarToFloat(float val) {
	return val;
}
[[nodiscard]] inline float ScalarToFloat(int val) {
	return static_cast<float>(val);
}

// Convert value to Expr<float>
template <typename T> [[nodiscard]] inline IR::Value::Expr<float> ToFloatExpr(T &&val) {
	using U = std::remove_cvref_t<T>;
	if constexpr (IsScalarType<U>::value) {
		return IR::Value::Expr<float>(ScalarToFloat(std::forward<T>(val)));
	} else {
		return IR::Value::Expr<float>(std::forward<T>(val));
	}
}
} // namespace Detail

// ============================================================================
// Broadcast construction - SINGLE SCALAR (Optimized)
// Generates vec3(x) instead of vec3(x, x, x) for GLSL native broadcast
// ============================================================================

// Single float broadcast - generates vec2(x)
template <typename X>
	requires Detail::FloatComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeFloat2(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::Vec2>(std::make_unique<IR::Node::LoadUniformNode>(std::format("vec2({})", xStr)));
}

// Single float broadcast - generates vec3(x) ✅ OPTIMIZED
template <typename X>
	requires Detail::FloatComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeFloat3(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::Vec3>(std::make_unique<IR::Node::LoadUniformNode>(std::format("vec3({})", xStr)));
}

// Single float broadcast - generates vec4(x)
template <typename X>
	requires Detail::FloatComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeFloat4(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::Vec4>(std::make_unique<IR::Node::LoadUniformNode>(std::format("vec4({})", xStr)));
}

// Single int broadcast
template <typename X>
	requires Detail::IntComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeInt2(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::IVec2>(std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec2({})", xStr)));
}

template <typename X>
	requires Detail::IntComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeInt3(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::IVec3>(std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec3({})", xStr)));
}

template <typename X>
	requires Detail::IntComponent<X> && (!Detail::IsVecType<std::remove_cvref_t<X>>::value)
[[nodiscard]] inline auto MakeInt4(X &&x) {
	std::string xStr = Detail::ToGLSLString(std::forward<X>(x));
	return IR::Value::Expr<Math::IVec4>(std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec4({})", xStr)));
}

// ============================================================================
// Component-wise construction (multiple arguments)
// ============================================================================

// Two components
template <typename X, typename Y>
	requires Detail::FloatComponent<X> && Detail::FloatComponent<Y>
[[nodiscard]] inline auto MakeFloat2(X &&x, Y &&y) {
	return Detail::MakeFloat2Impl(Detail::ToFloatExpr(std::forward<X>(x)), Detail::ToFloatExpr(std::forward<Y>(y)));
}

// Three components
template <typename X, typename Y, typename Z>
	requires Detail::FloatComponent<X> && Detail::FloatComponent<Y> && Detail::FloatComponent<Z>
[[nodiscard]] inline auto MakeFloat3(X &&x, Y &&y, Z &&z) {
	return Detail::MakeFloat3Impl(Detail::ToFloatExpr(std::forward<X>(x)), Detail::ToFloatExpr(std::forward<Y>(y)),
								  Detail::ToFloatExpr(std::forward<Z>(z)));
}

// Four components
template <typename X, typename Y, typename Z, typename W>
	requires Detail::FloatComponent<X> && Detail::FloatComponent<Y> && Detail::FloatComponent<Z> &&
			 Detail::FloatComponent<W>
[[nodiscard]] inline auto MakeFloat4(X &&x, Y &&y, Z &&z, W &&w) {
	return Detail::MakeFloat4Impl(Detail::ToFloatExpr(std::forward<X>(x)), Detail::ToFloatExpr(std::forward<Y>(y)),
								  Detail::ToFloatExpr(std::forward<Z>(z)), Detail::ToFloatExpr(std::forward<W>(w)));
}

// Int versions - component-wise
template <typename X, typename Y>
	requires Detail::IntComponent<X> && Detail::IntComponent<Y>
[[nodiscard]] inline auto MakeInt2(X &&x, Y &&y) {
	return Detail::MakeInt2Impl(IR::Value::Expr<int>(std::forward<X>(x)), IR::Value::Expr<int>(std::forward<Y>(y)));
}

template <typename X, typename Y, typename Z>
	requires Detail::IntComponent<X> && Detail::IntComponent<Y> && Detail::IntComponent<Z>
[[nodiscard]] inline auto MakeInt3(X &&x, Y &&y, Z &&z) {
	return Detail::MakeInt3Impl(IR::Value::Expr<int>(std::forward<X>(x)), IR::Value::Expr<int>(std::forward<Y>(y)),
								IR::Value::Expr<int>(std::forward<Z>(z)));
}

template <typename X, typename Y, typename Z, typename W>
	requires Detail::IntComponent<X> && Detail::IntComponent<Y> && Detail::IntComponent<Z> && Detail::IntComponent<W>
[[nodiscard]] inline auto MakeInt4(X &&x, Y &&y, Z &&z, W &&w) {
	return Detail::MakeInt4Impl(IR::Value::Expr<int>(std::forward<X>(x)), IR::Value::Expr<int>(std::forward<Y>(y)),
								IR::Value::Expr<int>(std::forward<Z>(z)), IR::Value::Expr<int>(std::forward<W>(w)));
}

// ============================================================================
// Vector swizzle broadcast (e.g., MakeFloat3(Vec2, float))
// ============================================================================

// MakeFloat3 from Vec2 + scalar
template <typename XY, typename Z>
	requires Detail::Vec2Component<XY> && Detail::FloatComponent<Z>
[[nodiscard]] inline auto MakeFloat3(XY &&xy, Z &&z) {
	std::string xyStr = Detail::ToGLSLString(std::forward<XY>(xy));
	std::string zStr  = Detail::ToGLSLString(std::forward<Z>(z));
	return IR::Value::Expr<Math::Vec3>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec3(({}).xy, {})", xyStr, zStr)));
}

// MakeFloat4 from Vec3 + scalar
template <typename XYZ, typename W>
	requires Detail::Vec3Component<XYZ> && Detail::FloatComponent<W>
[[nodiscard]] inline auto MakeFloat4(XYZ &&xyz, W &&w) {
	std::string xyzStr = Detail::ToGLSLString(std::forward<XYZ>(xyz));
	std::string wStr   = Detail::ToGLSLString(std::forward<W>(w));
	return IR::Value::Expr<Math::Vec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec4(({}).xyz, {})", xyzStr, wStr)));
}

// MakeFloat4 from Vec2 + scalar + scalar
template <typename XY, typename Z, typename W>
	requires Detail::Vec2Component<XY> && Detail::FloatComponent<Z> && Detail::FloatComponent<W>
[[nodiscard]] inline auto MakeFloat4(XY &&xy, Z &&z, W &&w) {
	std::string xyStr = Detail::ToGLSLString(std::forward<XY>(xy));
	std::string zStr  = Detail::ToGLSLString(std::forward<Z>(z));
	std::string wStr  = Detail::ToGLSLString(std::forward<W>(w));
	return IR::Value::Expr<Math::Vec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec4(({}).xy, {}, {})", xyStr, zStr, wStr)));
}

// Int broadcast versions
// MakeInt3 from IVec2 + scalar
template <typename XY, typename Z>
	requires Detail::IVec2Component<XY> && Detail::IntComponent<Z>
[[nodiscard]] inline auto MakeInt3(XY &&xy, Z &&z) {
	std::string xyStr = Detail::ToGLSLString(std::forward<XY>(xy));
	std::string zStr  = Detail::ToGLSLString(std::forward<Z>(z));
	return IR::Value::Expr<Math::IVec3>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec3(({}).xy, {})", xyStr, zStr)));
}

// MakeInt4 from IVec3 + scalar
template <typename XYZ, typename W>
	requires Detail::IVec3Component<XYZ> && Detail::IntComponent<W>
[[nodiscard]] inline auto MakeInt4(XYZ &&xyz, W &&w) {
	std::string xyzStr = Detail::ToGLSLString(std::forward<XYZ>(xyz));
	std::string wStr   = Detail::ToGLSLString(std::forward<W>(w));
	return IR::Value::Expr<Math::IVec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec4(({}).xyz, {})", xyzStr, wStr)));
}

// MakeInt4 from IVec2 + scalar + scalar
template <typename XY, typename Z, typename W>
	requires Detail::IVec2Component<XY> && Detail::IntComponent<Z> && Detail::IntComponent<W>
[[nodiscard]] inline auto MakeInt4(XY &&xy, Z &&z, W &&w) {
	std::string xyStr = Detail::ToGLSLString(std::forward<XY>(xy));
	std::string zStr  = Detail::ToGLSLString(std::forward<Z>(z));
	std::string wStr  = Detail::ToGLSLString(std::forward<W>(w));
	return IR::Value::Expr<Math::IVec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec4(({}).xy, {}, {})", xyStr, zStr, wStr)));
}

// ============================================================================
// CPU-side constant to GPU Expr conversions
// ============================================================================

[[nodiscard]] inline IR::Value::Expr<Math::Vec2> MakeFloat2(const Math::Vec2 &v) {
	return IR::Value::Expr<Math::Vec2>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec2({}, {})", v.x, v.y)));
}

[[nodiscard]] inline IR::Value::Expr<Math::Vec3> MakeFloat3(const Math::Vec3 &v) {
	return IR::Value::Expr<Math::Vec3>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec3({}, {}, {})", v.x, v.y, v.z)));
}

[[nodiscard]] inline IR::Value::Expr<Math::Vec4> MakeFloat4(const Math::Vec4 &v) {
	return IR::Value::Expr<Math::Vec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("vec4({}, {}, {}, {})", v.x, v.y, v.z, v.w)));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec2> MakeInt2(const Math::IVec2 &v) {
	return IR::Value::Expr<Math::IVec2>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec2({}, {})", v.x, v.y)));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec3> MakeInt3(const Math::IVec3 &v) {
	return IR::Value::Expr<Math::IVec3>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec3({}, {}, {})", v.x, v.y, v.z)));
}

[[nodiscard]] inline IR::Value::Expr<Math::IVec4> MakeInt4(const Math::IVec4 &v) {
	return IR::Value::Expr<Math::IVec4>(
		std::make_unique<IR::Node::LoadUniformNode>(std::format("ivec4({}, {}, {}, {})", v.x, v.y, v.z, v.w)));
}

// ============================================================================
// Matrix construction helpers
// ============================================================================

namespace Detail {
[[nodiscard]] inline IR::Value::Expr<Math::Mat2> MakeMat2Impl(const IR::Value::Expr<Math::Vec2> &c0,
															  const IR::Value::Expr<Math::Vec2> &c1) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<Math::Vec2> *>(&c0),
												const_cast<IR::Value::Expr<Math::Vec2> *>(&c1)};
	return Math::MakeCall<Math::Mat2>("mat2", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::Mat3> MakeMat3Impl(const IR::Value::Expr<Math::Vec3> &c0,
															  const IR::Value::Expr<Math::Vec3> &c1,
															  const IR::Value::Expr<Math::Vec3> &c2) {
	std::vector<IR::Value::ExprBase *> exprs = {const_cast<IR::Value::Expr<Math::Vec3> *>(&c0),
												const_cast<IR::Value::Expr<Math::Vec3> *>(&c1),
												const_cast<IR::Value::Expr<Math::Vec3> *>(&c2)};
	return Math::MakeCall<Math::Mat3>("mat3", BuildVectorParams(exprs));
}

[[nodiscard]] inline IR::Value::Expr<Math::Mat4> MakeMat4Impl(const IR::Value::Expr<Math::Vec4> &c0,
															  const IR::Value::Expr<Math::Vec4> &c1,
															  const IR::Value::Expr<Math::Vec4> &c2,
															  const IR::Value::Expr<Math::Vec4> &c3) {
	std::vector<IR::Value::ExprBase *> exprs = {
		const_cast<IR::Value::Expr<Math::Vec4> *>(&c0), const_cast<IR::Value::Expr<Math::Vec4> *>(&c1),
		const_cast<IR::Value::Expr<Math::Vec4> *>(&c2), const_cast<IR::Value::Expr<Math::Vec4> *>(&c3)};
	return Math::MakeCall<Math::Mat4>("mat4", BuildVectorParams(exprs));
}
} // namespace Detail

[[nodiscard]] inline auto MakeMat2(const IR::Value::Expr<Math::Vec2> &c0, const IR::Value::Expr<Math::Vec2> &c1) {
	return Detail::MakeMat2Impl(c0, c1);
}

[[nodiscard]] inline auto MakeMat3(const IR::Value::Expr<Math::Vec3> &c0, const IR::Value::Expr<Math::Vec3> &c1,
								   const IR::Value::Expr<Math::Vec3> &c2) {
	return Detail::MakeMat3Impl(c0, c1, c2);
}

[[nodiscard]] inline auto MakeMat4(const IR::Value::Expr<Math::Vec4> &c0, const IR::Value::Expr<Math::Vec4> &c1,
								   const IR::Value::Expr<Math::Vec4> &c2, const IR::Value::Expr<Math::Vec4> &c3) {
	return Detail::MakeMat4Impl(c0, c1, c2, c3);
}

// CPU-side matrix to GPU Expr conversions
[[nodiscard]] inline IR::Value::Expr<Math::Mat2> MakeMat2(const Math::Mat2 &m) {
	return IR::Value::Expr<Math::Mat2>(std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m)));
}

[[nodiscard]] inline IR::Value::Expr<Math::Mat3> MakeMat3(const Math::Mat3 &m) {
	return IR::Value::Expr<Math::Mat3>(std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m)));
}

[[nodiscard]] inline IR::Value::Expr<Math::Mat4> MakeMat4(const Math::Mat4 &m) {
	return IR::Value::Expr<Math::Mat4>(std::make_unique<IR::Node::LoadUniformNode>(IR::Value::ValueToString(m)));
}
} // namespace Construct

// ============================================================================
// Scalar construction helpers
// ============================================================================

template <typename T>
	requires std::convertible_to<std::remove_cvref_t<T>, float>
[[nodiscard]] inline auto MakeFloat(T &&x) {
	return IR::Value::Expr<float>(std::forward<T>(x));
}

template <typename T>
	requires std::convertible_to<std::remove_cvref_t<T>, int>
[[nodiscard]] inline auto MakeInt(T &&x) {
	return IR::Value::Expr<int>(std::forward<T>(x));
}

template <typename T>
	requires std::convertible_to<std::remove_cvref_t<T>, bool>
[[nodiscard]] inline auto MakeBool(T &&x) {
	return IR::Value::Expr<bool>(std::forward<T>(x));
}

// ============================================================================
// Convenience: bring Alias and Construct into GPU namespace
// ============================================================================
using namespace Alias;
using namespace Construct;
} // namespace GPU

#endif // EASYGPU_HELPERS_H
