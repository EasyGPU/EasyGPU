#pragma once

/**
 * @file Layers.h
 * @brief Neural network layer abstractions for EasyGPU AD training.
 *
 *   nn::Linear<T, InFeatures, OutFeatures>  — fully-connected layer
 *   nn::ReLU<T>, nn::Sigmoid<T>, nn::Tanh<T> — activation layers
 *   nn::Sequential<T, Layers...>             — compose layers into a pipeline
 *
 * Usage:
 *   // Outside kernel:
 *   Linear<float, 784, 128> fc1;
 *   ReLU<float> relu(128);
 *   Linear<float, 128, 10> fc2;
 *
 *   // Inside kernel lambda:
 *   fc1.Setup(); fc2.Setup(); relu.Setup();
 *   fc1.Forward(inputBuf, id, hiddenBuf);
 *   relu.Forward(hiddenBuf, id, hiddenBuf);
 *   fc2.Forward(hiddenBuf, id, outputBuf);
 */

#ifndef EASYGPU_NN_LAYERS_H
#define EASYGPU_NN_LAYERS_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

namespace GPU::NN {

// =============================================================================
// Weight initialization helper
// =============================================================================
namespace detail {

inline float XavierInit(unsigned &seed, size_t fanIn, size_t fanOut) {
	// Xavier uniform: ± sqrt(6 / (fanIn + fanOut))
	float range = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
	// Simple LCG for deterministic init
	seed		= seed * 1664525u + 1013904223u;
	float r		= static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
	return (r * 2.0f - 1.0f) * range;
}

} // namespace detail

// =============================================================================
// Linear — fully-connected layer
// =============================================================================

template <typename T, size_t InFeatures, size_t OutFeatures> class Linear {
	static_assert(std::is_same_v<T, float>, "Linear only supports float");

public:
	/** Construct with Xavier-initialized weights and zero biases. */
	Linear(unsigned initSeed = 42) {
		std::vector<T> wData(InFeatures * OutFeatures);
		unsigned	   seed = initSeed;
		for (size_t j = 0; j < OutFeatures; j++) {
			for (size_t i = 0; i < InFeatures; i++) {
				wData[j * InFeatures + i] = detail::XavierInit(seed, InFeatures, OutFeatures);
			}
		}
		weight_ = Tensor<T, InFeatures, OutFeatures>(wData);
		bias_	= Tensor<T, OutFeatures>(std::vector<T>(OutFeatures, T{}));
	}

	/** Re-initialize weights (useful for retraining). */
	void Reset(unsigned initSeed = 42) {
		unsigned seed = initSeed;
		for (size_t j = 0; j < OutFeatures; j++)
			for (size_t i = 0; i < InFeatures; i++)
				weight_(i, j) = detail::XavierInit(seed, InFeatures, OutFeatures);
		for (size_t o = 0; o < OutFeatures; o++)
			bias_(o) = T{};
		weight_.Upload();
		bias_.Upload();
	}

	/**
	 * Bind weight and bias tensors and register all AD parameters.
	 * Must be called inside the kernel lambda during ADKernel construction.
	 */
	void Setup() {
		weightRef_ = weight_.Bind();
		biasRef_   = bias_.Bind();
		weightRef_.RegisterAsParam();
		biasRef_.RegisterAsParam();
	}

	/**
	 * Forward pass: output[threadId][o] = sum_i weight[o][i] * input[threadId][i] + bias[o].
	 * Uses DSL For loops — works for any compile-time feature dimensions.
	 */
	void Forward(const IR::Value::BufferRef<T> &input, const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &output) {
		constexpr int kIn  = static_cast<int>(InFeatures);
		constexpr int kOut = static_cast<int>(OutFeatures);

		GPU::Flow::For(MakeInt(0), MakeInt(kOut), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			sum					  = biasRef_[o];
			GPU::Flow::For(MakeInt(0), MakeInt(kIn), [&](IR::Value::Var<int> &i) {
				IR::Value::Var<T> prod = weightRef_(o, i) * input[threadId * kIn + i];
				sum					   = sum + prod;
			});
			output[threadId * kOut + o] = sum;
		});
	}

	// ---- Data access (for optimizer registration) ----

	Tensor<T, InFeatures, OutFeatures> &Weight() {
		return weight_;
	}
	Tensor<T, OutFeatures> &Bias() {
		return bias_;
	}
	const Tensor<T, InFeatures, OutFeatures> &Weight() const {
		return weight_;
	}
	const Tensor<T, OutFeatures> &Bias() const {
		return bias_;
	}

	T *WeightData() {
		return weight_.Data();
	}
	T *BiasData() {
		return bias_.Data();
	}

	static constexpr size_t InputDim() {
		return InFeatures;
	}
	static constexpr size_t OutputDim() {
		return OutFeatures;
	}
	size_t ParamCount() const {
		return InFeatures * OutFeatures + OutFeatures;
	}

private:
	Tensor<T, InFeatures, OutFeatures>	  weight_;
	Tensor<T, OutFeatures>				  bias_;
	TensorRef<T, InFeatures, OutFeatures> weightRef_;
	TensorRef<T, OutFeatures>			  biasRef_;
};

// =============================================================================
// ReLU — rectified linear unit
// =============================================================================

template <typename T = float> class ReLU {
public:
	explicit ReLU(size_t numElements = 0) : numElements_(numElements) {
	}

	void Setup() {
	} // no-op for stateless activations

	/**
	 * Element-wise ReLU: out[i] = max(in[i], 0).
	 * Supports in-place when in and out refer to the same buffer.
	 */
	void Forward(const IR::Value::BufferRef<T> &in, const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &out) {
		int n = static_cast<int>(numElements_);
		GPU::Flow::For(MakeInt(0), MakeInt(n), [&](IR::Value::Var<int> &i) {
			IR::Value::Var<T> v	  = in[threadId * n + i];
			out[threadId * n + i] = GPU::Math::Max(v, MakeFloat(0.0f));
		});
	}

private:
	size_t numElements_;
};

// =============================================================================
// Sigmoid — logistic activation
// =============================================================================

template <typename T = float> class Sigmoid {
public:
	explicit Sigmoid(size_t numElements = 0) : numElements_(numElements) {
	}

	void Setup() {
	}

	void Forward(const IR::Value::BufferRef<T> &in, const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &out) {
		int n = static_cast<int>(numElements_);
		GPU::Flow::For(MakeInt(0), MakeInt(n), [&](IR::Value::Var<int> &i) {
			IR::Value::Var<T> v	  = in[threadId * n + i];
			out[threadId * n + i] = MakeFloat(1.0f) / (MakeFloat(1.0f) + GPU::Math::Exp(MakeFloat(0.0f) - v));
		});
	}

private:
	size_t numElements_;
};

// =============================================================================
// TanhActivation — hyperbolic tangent activation
// =============================================================================

template <typename T = float> class TanhActivation {
public:
	explicit TanhActivation(size_t numElements = 0) : numElements_(numElements) {
	}

	void Setup() {
	}

	void Forward(const IR::Value::BufferRef<T> &in, const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &out) {
		int n = static_cast<int>(numElements_);
		GPU::Flow::For(MakeInt(0), MakeInt(n), [&](IR::Value::Var<int> &i) {
			IR::Value::Var<T> v	   = in[threadId * n + i];
			// tanh(x) = 2*sigmoid(2x) - 1
			IR::Value::Var<T> twoX = v * MakeFloat(2.0f);
			out[threadId * n + i] =
				MakeFloat(2.0f) / (MakeFloat(1.0f) + GPU::Math::Exp(MakeFloat(0.0f) - twoX)) - MakeFloat(1.0f);
		});
	}

private:
	size_t numElements_;
};

// =============================================================================
// Sequential — compose layers into a pipeline
// =============================================================================

template <typename T, typename... Layers> class Sequential {
public:
	/**
	 * @param batchSize    Number of data samples (GPU threads).
	 * @param maxInterDim  Maximum intermediate dimension across layers
	 *                     (e.g., for Linear<784,128>→ReLU→Linear<128,10>, maxInterDim=128).
	 */
	Sequential(size_t batchSize, size_t maxInterDim)
		: batchSize_(batchSize), interDim_(maxInterDim),
		  interBufA_(batchSize * maxInterDim, Runtime::BufferMode::ReadWrite),
		  interBufB_(batchSize * maxInterDim, Runtime::BufferMode::ReadWrite) {
	}

	/** Bind all layer parameters and the internal intermediate buffer. */
	void Setup() {
		interRefA_ = interBufA_.Bind();
		interRefB_ = interBufB_.Bind();
		std::apply([](auto &...layer) { (layer.Setup(), ...); }, layers_);
	}

	/**
	 * Forward pass through all layers sequentially.
	 * Uses the internal intermediate buffer between layers.
	 */
	void Forward(const IR::Value::BufferRef<T> &input, const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &output) {
		ForwardStep<0>(input, threadId, output);
	}

	/** Access a layer by index. */
	template <size_t I> auto &Get() {
		return std::get<I>(layers_);
	}

	template <size_t I> const auto &Get() const {
		return std::get<I>(layers_);
	}

	/** Number of layers. */
	static constexpr size_t NumLayers = sizeof...(Layers);

private:
	template <size_t I>
	void ForwardStep(const IR::Value::BufferRef<T> &in, const IR::Value::Var<int> &threadId,
					 const IR::Value::BufferRef<T> &finalOut) {
		auto &layer = std::get<I>(layers_);
		if constexpr (I == sizeof...(Layers) - 1) {
			// Last layer — output goes to final destination
			layer.Forward(in, threadId, finalOut);
		} else {
			// Alternate intermediate buffers so non-in-place layers never read
			// and write overlapping addresses in the same forward step.
			auto &out = InterRef<I>();
			layer.Forward(in, threadId, out);
			ForwardStep<I + 1>(out, threadId, finalOut);
		}
	}

	template <size_t I> IR::Value::BufferRef<T> &InterRef() {
		if constexpr (I % 2 == 0)
			return interRefA_;
		else
			return interRefB_;
	}

	std::tuple<Layers...>	layers_;
	size_t					batchSize_;
	size_t					interDim_;
	Runtime::Buffer<T>		interBufA_;
	Runtime::Buffer<T>		interBufB_;
	IR::Value::BufferRef<T> interRefA_;
	IR::Value::BufferRef<T> interRefB_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_LAYERS_H
