#pragma once

/**
 * @file FusedMLP.h
 * @brief Single-kernel MLP blocks that keep hidden activations in shader locals.
 */

#ifndef EASYGPU_NN_FUSED_MLP_H
#define EASYGPU_NN_FUSED_MLP_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>
#include <Utility/Math.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace GPU::NN {

enum class FusedActivation {
	ReLU,
	Tanh,
	None
};

namespace detail {

inline float FusedXavier(unsigned &seed, size_t fanIn, size_t fanOut) {
	seed = seed * 1664525u + 1013904223u;
	float u = static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
	float range = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
	return (u * 2.0f - 1.0f) * range;
}

template <typename T>
IR::Value::Var<T> ApplyFusedActivation(const IR::Value::Var<T> &x, FusedActivation activation) {
	switch (activation) {
	case FusedActivation::ReLU:
		return GPU::Math::Max(x, MakeFloat(0.0f));
	case FusedActivation::Tanh:
		return MakeFloat(2.0f) /
			(MakeFloat(1.0f) + GPU::Math::Exp(MakeFloat(0.0f) - x * MakeFloat(2.0f)))
			- MakeFloat(1.0f);
	case FusedActivation::None:
	default:
		return x;
	}
}

} // namespace detail

// =============================================================================
// FusedMLP2
// =============================================================================

template <typename T, size_t InFeatures, size_t HiddenFeatures, size_t OutFeatures>
class FusedMLP2 {
	static_assert(std::is_same_v<T, float>, "FusedMLP2 only supports float");

public:
	FusedMLP2(unsigned initSeed = 42,
			  FusedActivation activation = FusedActivation::ReLU)
		: activation_(activation) {
		Reset(initSeed);
	}

	void Reset(unsigned initSeed = 42) {
		unsigned seed = initSeed;
		std::vector<T> w1(HiddenFeatures * InFeatures);
		std::vector<T> b1(HiddenFeatures, T{});
		std::vector<T> w2(OutFeatures * HiddenFeatures);
		std::vector<T> b2(OutFeatures, T{});

		for (size_t h = 0; h < HiddenFeatures; h++)
			for (size_t i = 0; i < InFeatures; i++)
				w1[h * InFeatures + i] = detail::FusedXavier(seed, InFeatures, HiddenFeatures);

		for (size_t o = 0; o < OutFeatures; o++)
			for (size_t h = 0; h < HiddenFeatures; h++)
				w2[o * HiddenFeatures + h] = detail::FusedXavier(seed, HiddenFeatures, OutFeatures);

		w1_ = Tensor<T, HiddenFeatures, InFeatures>(w1);
		b1_ = Tensor<T, HiddenFeatures>(b1);
		w2_ = Tensor<T, OutFeatures, HiddenFeatures>(w2);
		b2_ = Tensor<T, OutFeatures>(b2);
	}

	void Setup(bool registerParams = true) {
		w1Ref_ = w1_.Bind();
		b1Ref_ = b1_.Bind();
		w2Ref_ = w2_.Bind();
		b2Ref_ = b2_.Bind();

		if (registerParams) {
			w1Ref_.ForEachParam([](auto &p) { AD::Param(p); });
			b1Ref_.ForEachParam([](auto &p) { AD::Param(p); });
			w2Ref_.ForEachParam([](auto &p) { AD::Param(p); });
			b2Ref_.ForEachParam([](auto &p) { AD::Param(p); });
		}
	}

	void Forward(const IR::Value::BufferRef<T> &input,
				 const IR::Value::Var<int> &threadId,
				 const IR::Value::BufferRef<T> &output) {
		std::array<IR::Value::Var<T>, HiddenFeatures> hidden;

		for (size_t h = 0; h < HiddenFeatures; h++) {
			IR::Value::Var<T> sum = b1Ref_[static_cast<int>(h)];
			for (size_t i = 0; i < InFeatures; i++) {
				sum = sum + w1Ref_(static_cast<int>(h), static_cast<int>(i))
					* input[threadId * static_cast<int>(InFeatures) + static_cast<int>(i)];
			}
			hidden[h] = detail::ApplyFusedActivation(sum, activation_);
		}

		for (size_t o = 0; o < OutFeatures; o++) {
			IR::Value::Var<T> sum = b2Ref_[static_cast<int>(o)];
			for (size_t h = 0; h < HiddenFeatures; h++) {
				sum = sum + w2Ref_(static_cast<int>(o), static_cast<int>(h)) * hidden[h];
			}
			output[threadId * static_cast<int>(OutFeatures) + static_cast<int>(o)] = sum;
		}
	}

	Tensor<T, HiddenFeatures, InFeatures> &W1() { return w1_; }
	Tensor<T, HiddenFeatures> &B1() { return b1_; }
	Tensor<T, OutFeatures, HiddenFeatures> &W2() { return w2_; }
	Tensor<T, OutFeatures> &B2() { return b2_; }

	static constexpr size_t ParamCount() {
		return HiddenFeatures * InFeatures + HiddenFeatures
			+ OutFeatures * HiddenFeatures + OutFeatures;
	}

private:
	FusedActivation activation_;
	Tensor<T, HiddenFeatures, InFeatures> w1_;
	Tensor<T, HiddenFeatures> b1_;
	Tensor<T, OutFeatures, HiddenFeatures> w2_;
	Tensor<T, OutFeatures> b2_;

	TensorRef<T, HiddenFeatures, InFeatures> w1Ref_;
	TensorRef<T, HiddenFeatures> b1Ref_;
	TensorRef<T, OutFeatures, HiddenFeatures> w2Ref_;
	TensorRef<T, OutFeatures> b2Ref_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_FUSED_MLP_H
