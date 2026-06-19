#pragma once

/**
 * @file AdjointInspector.h
 * @brief User-facing AD inspector that generates forward + backward GLSL.
 *
 * AdjointInspector1D works like InspectorKernel1D but also records the
 * gradient tape and generates the backward-pass GLSL code. This is the
 * primary API for inspecting and debugging automatic differentiation.
 *
 * Usage:
 *   AdjointInspector1D inspector([](Var<int>& id, auto& ctx) {
 *       Var<float> w; w = 2.0f;
 *       Var<float> x; x = 3.0f;
 *       Var<float> y = w * x;
 *       ctx.MarkLoss(y, "float");
 *   });
 *   std::cout << inspector.GetForwardCode();
 *   std::cout << inspector.GetBackwardCode();
 *   inspector.PrintTape();
 */

#ifndef EASYGPU_AD_ADJOINTINSPECTOR_H
#define EASYGPU_AD_ADJOINTINSPECTOR_H

#include <AD/AdjointGenerator.h>
#include <AD/GradientTape.h>

#include <Kernel/Kernel.h>

#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace GPU::AD {

/**
 * Context passed to the user's kernel function for AD configuration.
 *
 * Provides methods to register trainable parameters and mark the loss
 * variable. The context is valid only during kernel construction.
 */
class AdjointContext {
public:
	AdjointContext(GradientTape &tape) : _tape(tape) {
	}

	/**
	 * Register a variable as a trainable parameter by type-deduced Var<T>.
	 * This is the recommended API — no strings, no .template noise.
	 *
	 *     ctx.RegisterParameter(w);     // Var<float> → "float"
	 *     ctx.RegisterParameter(v);     // Var<Vec3> → "vec3"
	 */
	template <typename T> void RegisterParameter(const IR::Value::Var<T> &var) {
		_tape.RegisterParameter(var.VarName(), TypeName<T>());
	}

	/**
	 * Mark a variable as the scalar loss by type-deduced Var<T>.
	 *
	 *     ctx.MarkLoss(loss);           // Var<float> → "float"
	 */
	template <typename T> void MarkLoss(const IR::Value::Var<T> &var) {
		_tape.MarkLoss(var.VarName(), TypeName<T>());
	}

	// ---- String-based overloads (for dynamic-name or legacy code) ----

	/** Register a parameter by explicit name and GLSL type string. */
	void RegisterParameter(const std::string &name, const std::string &glslType) {
		_tape.RegisterParameter(name, glslType);
	}

	/** Register a parameter by name with explicit template type. */
	template <typename T> void RegisterParameter(const std::string &name) {
		_tape.RegisterParameter(name, TypeName<T>());
	}

	/** Mark a loss variable by explicit name and GLSL type string. */
	void MarkLoss(const std::string &name, const std::string &glslType) {
		_tape.MarkLoss(name, glslType);
	}

	/** Mark a loss variable by name with explicit template type. */
	template <typename T> void MarkLoss(const std::string &name) {
		_tape.MarkLoss(name, TypeName<T>());
	}

	/** Get the underlying gradient tape. */
	GradientTape &Tape() {
		return _tape;
	}

private:
	/** Map C++ types to GLSL type strings. */
	template <typename T> static std::string TypeName() {
		if constexpr (std::is_same_v<T, float>)
			return "float";
		else if constexpr (std::is_same_v<T, int>)
			return "int";
		else if constexpr (std::is_same_v<T, bool>)
			return "bool";
		else if constexpr (std::is_same_v<T, Math::Vec2>)
			return "vec2";
		else if constexpr (std::is_same_v<T, Math::Vec3>)
			return "vec3";
		else if constexpr (std::is_same_v<T, Math::Vec4>)
			return "vec4";
		else if constexpr (std::is_same_v<T, Math::IVec2>)
			return "ivec2";
		else if constexpr (std::is_same_v<T, Math::IVec3>)
			return "ivec3";
		else if constexpr (std::is_same_v<T, Math::IVec4>)
			return "ivec4";
		else
			return "float";
	}

	GradientTape &_tape;
};

/**
 * 1D AD inspector kernel.
 *
 * Constructs a forward-pass kernel from a DSL lambda, records the gradient
 * tape, and generates the backward-pass GLSL code. Works like
 * InspectorKernel1D but adds automatic differentiation support.
 *
 * The kernel function signature is:
 *   void(Var<int>& threadId, AdjointContext& ctx)
 *
 * @tparam Func The kernel function type
 */
template <typename Func> class AdjointInspector1D {
public:
	/**
	 * Construct the AD inspector kernel.
	 *
	 * @param kernelFunc The DSL kernel function
	 * @param workSizeX The work group size in X dimension (default 256)
	 */
	AdjointInspector1D(Func &&kernelFunc, int workSizeX = 256) : _workSizeX(workSizeX) {

		// Phase 1: Forward pass — record to tape while building GLSL
		{
			auto									&builder = IR::Builder::Builder::Get();
			IR::Builder::Builder::ScopedGradientTape tapeGuard(builder, &_tape);

			_forwardKernel = std::make_unique<Kernel::InspectorKernel1D>(
				[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &id) {
					AdjointContext ctx(_tape);
					func(id, ctx);
				},
				workSizeX);
		}

		// Phase 2: Backward pass — generate adjoint GLSL
		_backwardCode = _generator.Generate(_tape, true);
	}

	/** Get the forward-pass GLSL code. */
	std::string GetForwardCode() const {
		return _forwardKernel->GetCode();
	}

	/** Get the backward-pass GLSL code. */
	std::string GetBackwardCode() const {
		return _backwardCode;
	}

	/** Get the tape summary (for debugging). */
	std::string GetTapeSummary() const {
		std::string summary;
		for (size_t i = 0; i < _tape.Size(); ++i) {
			const auto &e  = _tape[i];
			summary		  += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
			if (!e.intrinsicName.empty())
				summary += " fn=" + e.intrinsicName;
			summary += " ins:";
			for (const auto &in : e.inputs)
				summary += in.name + ",";
			summary += "\n";
		}
		return summary;
	}

	/** Print the tape summary to stdout. */
	void PrintTape() {
		std::cout << GetTapeSummary();
	}

	/** Get the underlying gradient tape. */
	const GradientTape &Tape() const {
		return _tape;
	}

	/** Get the adjoint table after generation. */
	const AdjointTable &Adjoints() const {
		return _generator.GetAdjointTable();
	}

	/** Check if the backward code was generated successfully. */
	bool HasBackwardCode() const {
		return !_backwardCode.empty();
	}

	/** Get the work group size. */
	int WorkSizeX() const {
		return _workSizeX;
	}

private:
	int										   _workSizeX;
	GradientTape							   _tape;
	AdjointGenerator						   _generator;
	std::unique_ptr<Kernel::InspectorKernel1D> _forwardKernel;
	std::string								   _backwardCode;
};

/**
 * 2D AD inspector kernel.
 *
 * The kernel function signature is:
 *   void(Var<int>& idX, Var<int>& idY, AdjointContext& ctx)
 *
 * @tparam Func The kernel function type
 */
template <typename Func> class AdjointInspector2D {
public:
	AdjointInspector2D(Func &&kernelFunc, int workSizeX = 16, int workSizeY = 16)
		: _workSizeX(workSizeX), _workSizeY(workSizeY) {

		{
			auto									&builder = IR::Builder::Builder::Get();
			IR::Builder::Builder::ScopedGradientTape tapeGuard(builder, &_tape);

			_forwardKernel = std::make_unique<Kernel::InspectorKernel2D>(
				[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &idX, IR::Value::Var<int> &idY) {
					AdjointContext ctx(_tape);
					func(idX, idY, ctx);
				},
				workSizeX, workSizeY);
		}

		_backwardCode = _generator.Generate(_tape, true);
	}

	std::string GetForwardCode() const {
		return _forwardKernel->GetCode();
	}
	std::string GetBackwardCode() const {
		return _backwardCode;
	}
	std::string GetTapeSummary() const {
		std::string summary;
		for (size_t i = 0; i < _tape.Size(); ++i) {
			const auto &e  = _tape[i];
			summary		  += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
			if (!e.intrinsicName.empty())
				summary += " fn=" + e.intrinsicName;
			summary += " ins:";
			for (const auto &in : e.inputs)
				summary += in.name + ",";
			summary += "\n";
		}
		return summary;
	}
	void PrintTape() {
		std::cout << GetTapeSummary();
	}
	bool HasBackwardCode() const {
		return !_backwardCode.empty();
	}

private:
	int										   _workSizeX, _workSizeY;
	GradientTape							   _tape;
	AdjointGenerator						   _generator;
	std::unique_ptr<Kernel::InspectorKernel2D> _forwardKernel;
	std::string								   _backwardCode;
};

/**
 * 3D AD inspector kernel.
 *
 * The kernel function signature is:
 *   void(Var<int>& idX, Var<int>& idY, Var<int>& idZ, AdjointContext& ctx)
 *
 * @tparam Func The kernel function type
 */
template <typename Func> class AdjointInspector3D {
public:
	AdjointInspector3D(Func &&kernelFunc, int workSizeX = 8, int workSizeY = 8, int workSizeZ = 4)
		: _workSizeX(workSizeX), _workSizeY(workSizeY), _workSizeZ(workSizeZ) {

		{
			auto									&builder = IR::Builder::Builder::Get();
			IR::Builder::Builder::ScopedGradientTape tapeGuard(builder, &_tape);

			_forwardKernel = std::make_unique<Kernel::InspectorKernel3D>(
				[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &idX, IR::Value::Var<int> &idY,
															  IR::Value::Var<int> &idZ) {
					AdjointContext ctx(_tape);
					func(idX, idY, idZ, ctx);
				},
				workSizeX, workSizeY, workSizeZ);
		}

		_backwardCode = _generator.Generate(_tape, true);
	}

	std::string GetForwardCode() const {
		return _forwardKernel->GetCode();
	}
	std::string GetBackwardCode() const {
		return _backwardCode;
	}
	std::string GetTapeSummary() const {
		std::string summary;
		for (size_t i = 0; i < _tape.Size(); ++i) {
			const auto &e  = _tape[i];
			summary		  += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
			if (!e.intrinsicName.empty())
				summary += " fn=" + e.intrinsicName;
			summary += " ins:";
			for (const auto &in : e.inputs)
				summary += in.name + ",";
			summary += "\n";
		}
		return summary;
	}
	void PrintTape() {
		std::cout << GetTapeSummary();
	}
	bool HasBackwardCode() const {
		return !_backwardCode.empty();
	}

private:
	int										   _workSizeX, _workSizeY, _workSizeZ;
	GradientTape							   _tape;
	AdjointGenerator						   _generator;
	std::unique_ptr<Kernel::InspectorKernel3D> _forwardKernel;
	std::string								   _backwardCode;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADJOINTINSPECTOR_H
