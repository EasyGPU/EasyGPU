#pragma once

/**
 * @file Attention.h
 * @brief Causal multi-head self-attention for EasyGPU GPT-style transformers.
 *
 *   CausalSelfAttention<T, EmbedDim, NumHeads>
 *
 * Uses three-pass softmax (max reduction, exp-sum, weighted sum).
 * No biases (GPT-2 style). All Q/K/V/O weights packed into a single 3D tensor
 * to minimise GPU SSBO binding slots.
 */

#ifndef EASYGPU_NN_ATTENTION_H
#define EASYGPU_NN_ATTENTION_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace GPU::NN {

template <typename T, size_t EmbedDim, size_t NumHeads>
class CausalSelfAttention {
	static_assert(std::is_same_v<T, float>, "CausalSelfAttention only supports float");
	static_assert(EmbedDim % NumHeads == 0, "EmbedDim must be divisible by NumHeads");
	static constexpr size_t HeadDim = EmbedDim / NumHeads;

	// Weight tensor layout: [4][EmbedDim][EmbedDim]
	//   layer 0 = Wq, layer 1 = Wk, layer 2 = Wv, layer 3 = Wo
	static constexpr size_t NUM_LAYERS = 4;

public:
	CausalSelfAttention(unsigned initSeed = 42) {
		std::vector<T> wData(NUM_LAYERS * EmbedDim * EmbedDim);
		float range = std::sqrt(6.0f / static_cast<float>(2 * EmbedDim));

		for (size_t l = 0; l < NUM_LAYERS; l++) {
			unsigned s = initSeed + l;
			for (size_t j = 0; j < EmbedDim; j++) {
				for (size_t i = 0; i < EmbedDim; i++) {
					s = s * 1664525u + 1013904223u;
					size_t idx = l * EmbedDim * EmbedDim + j * EmbedDim + i;
					wData[idx] = (static_cast<float>(s) / UINT32_MAX * 2.0f - 1.0f) * range;
				}
			}
		}
		w_ = Tensor<T, NUM_LAYERS, EmbedDim, EmbedDim>(wData);
	}

	void Setup() {
		wRef_ = w_.Bind();
		wRef_.ForEachParam([](auto &p) { AD::Param(p); });
	}

	/**
	 * Forward pass for a single position in a sequence.
	 *
	 * @param scratch Single scratch buffer holding all regions
	 * @param xOff    Base offset of input (normalized residual) within scratch
	 * @param kOff    Base offset of K region within scratch
	 * @param vOff    Base offset of V region within scratch
	 * @param aOff    Base offset of AttnOut region within scratch
	 * @param pos     Current position (0..blockSize-1)
	 * @param offset  Base offset for this batch = batchIdx * blockSize * embedDim
	 */
	void Forward(const IR::Value::BufferRef<T> &scratch,
				 const IR::Value::Expr<int> &xOff,
				 const IR::Value::Expr<int> &kOff,
				 const IR::Value::Expr<int> &vOff,
				 const IR::Value::Expr<int> &aOff,
				 const IR::Value::Expr<int> &dotsOff,
				 const IR::Value::Var<int> &pos,
				 const IR::Value::Expr<int> &offset) {
		constexpr int E = static_cast<int>(EmbedDim);
		constexpr int H = static_cast<int>(NumHeads);
		constexpr int HD = static_cast<int>(HeadDim);

		IR::Value::Expr<int> po = offset + pos * E;
		IR::Value::Expr<int> dotsBase = dotsOff + offset * MakeInt(4);

		// --- Q/K/V projections for current position ---
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sq = MakeFloat(0.0f);
			IR::Value::Var<T> sk = MakeFloat(0.0f);
			IR::Value::Var<T> sv = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				IR::Value::Var<T> xi = scratch[xOff + po + i];
				IR::Value::Var<T> p0 = wRef_(0, o, i) * xi;
				IR::Value::Var<T> p1 = wRef_(1, o, i) * xi;
				IR::Value::Var<T> p2 = wRef_(2, o, i) * xi;
				sq = sq + p0;
				sk = sk + p1;
				sv = sv + p2;
			});
			scratch[kOff + po + o] = sk;
			scratch[vOff + po + o] = sv;
			scratch[aOff + po + o] = sq;
		});

		// --- Safe softmax: 3-pass with max, dot computed ONCE (stored) ---
		IR::Value::Expr<int> endPos = pos + MakeInt(1);

		GPU::Flow::For(MakeInt(0), MakeInt(H), [&](IR::Value::Var<int> &h) {
			IR::Value::Expr<int> hs = h * MakeInt(HD);

			// Pass 1: compute dots, find max, store dots in unused part of scratch
			// Store dots at kOff (Key region, above current position data)
			IR::Value::Var<T> maxLogit = MakeFloat(-1e9f);
			GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
				IR::Value::Expr<int> to = offset + t * E;
				IR::Value::Var<T> dot = MakeFloat(0.0f);
				GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &d) {
					IR::Value::Var<T> qk = scratch[aOff + po + hs + d] * scratch[kOff + to + hs + d];
					dot = dot + qk;
				});
				dot = dot / MakeFloat(std::sqrt(static_cast<float>(HD)));
				scratch[dotsBase + h * endPos + t] = dot;
				maxLogit = GPU::Math::Max(maxLogit, dot);
			});

			// Pass 2: exp(dot - max) and sum
			IR::Value::Var<T> sumExp = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
				IR::Value::Var<T> dot_t = scratch[dotsBase + h * endPos + t];
				IR::Value::Var<T> dm = dot_t - maxLogit;
				IR::Value::Var<T> edm = GPU::Math::Exp(dm);
				sumExp = sumExp + edm;
			});

			// Pass 3: weighted sum
			GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &d) {
				IR::Value::Var<T> sumV = MakeFloat(0.0f);
				GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
					IR::Value::Expr<int> to = offset + t * E;
					IR::Value::Var<T> dot_t2 = scratch[dotsBase + h * endPos + t];
					IR::Value::Var<T> dm2 = dot_t2 - maxLogit;
					IR::Value::Var<T> edm2 = GPU::Math::Exp(dm2);
					IR::Value::Var<T> weight = edm2 / sumExp;
					IR::Value::Var<T> wv = weight * scratch[vOff + to + hs + d];
					sumV = sumV + wv;
				});
				scratch[aOff + po + hs + d] = sumV;
			});
		});

		// --- Output projection ---
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				IR::Value::Var<T> wo = wRef_(3, o, i) * scratch[aOff + po + i];
				sum = sum + wo;
			});
			scratch[aOff + po + o] = sum;
		});
	}

	Tensor<T, NUM_LAYERS, EmbedDim, EmbedDim> &Weights() { return w_; }
	const Tensor<T, NUM_LAYERS, EmbedDim, EmbedDim> &Weights() const { return w_; }
	static constexpr size_t TotalSize = NUM_LAYERS * EmbedDim * EmbedDim;

private:
	Tensor<T, NUM_LAYERS, EmbedDim, EmbedDim> w_;
	TensorRef<T, NUM_LAYERS, EmbedDim, EmbedDim> wRef_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_ATTENTION_H
