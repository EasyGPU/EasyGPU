#pragma once

/**
 * @file Normalization.h
 * @brief RMS (Root Mean Square) normalization for EasyGPU NN.
 *
 *   RMSNorm<T, EmbedDim> — stateless RMS normalization (used in GPT-style transformers)
 *
 * Computes:  ms = mean(x_i^2),  scale = 1 / sqrt(ms + eps),  out_i = in_i * scale
 */

#ifndef EASYGPU_NN_NORMALIZATION_H
#define EASYGPU_NN_NORMALIZATION_H

#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>

#include <Flow/ForFlow.h>

#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <cstddef>

namespace GPU::NN {

template <typename T = float, size_t EmbedDim = 0>
class RMSNorm {
public:
	explicit RMSNorm(float eps = 1e-5f) : eps_(eps) {}

	void Setup() {} // stateless

	/**
	 * Apply RMS normalization in-place or to output buffer.
	 * @param in     Input buffer
	 * @param out    Output buffer (may alias in for in-place)
	 * @param offset Starting offset into both buffers
	 */
	void Forward(const IR::Value::BufferRef<T> &in,
				 const IR::Value::BufferRef<T> &out,
				 const IR::Value::Expr<int> &offset) {
		constexpr int N = static_cast<int>(EmbedDim);

		// Compute mean square
		IR::Value::Var<float> ms = MakeFloat(0.0f);
		GPU::Flow::For(MakeInt(0), MakeInt(N), [&](IR::Value::Var<int> &d) {
			IR::Value::Var<T> v = in[offset + d];
			IR::Value::Var<T> v2 = v * v;
			ms = ms + v2;
		});
		ms = ms / MakeFloat(static_cast<float>(EmbedDim));

		// RMS scale: (ms + eps)^(-0.5) = 1 / sqrt(ms + eps)
		IR::Value::Var<float> scale = GPU::Math::Pow(ms + MakeFloat(eps_), MakeFloat(-0.5f));

		// Apply normalization
		GPU::Flow::For(MakeInt(0), MakeInt(N), [&](IR::Value::Var<int> &d) {
			out[offset + d] = in[offset + d] * scale;
		});
	}

private:
	float eps_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_NORMALIZATION_H
