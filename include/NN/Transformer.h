#pragma once

/**
 * @file Transformer.h
 * @brief Transformer block (attention + MLP) with RMSNorm and residual connections.
 *
 *   TransformerBlock<T, BlockSize, EmbedDim, NumHeads>
 *
 * Pre-norm architecture (GPT-2 style): RMSNorm → Attention → +residual
 *                                     → RMSNorm → MLP (ReLU) → +residual
 */

#ifndef EASYGPU_NN_TRANSFORMER_H
#define EASYGPU_NN_TRANSFORMER_H

#include <AD/ADKernel.h>
#include <NN/Attention.h>
#include <NN/Normalization.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <cstddef>
#include <random>
#include <vector>

namespace GPU::NN {

template <typename T, size_t BlockSize, size_t EmbedDim, size_t NumHeads>
class TransformerBlock {
	static_assert(std::is_same_v<T, float>, "TransformerBlock only supports float");
	static_assert(EmbedDim % NumHeads == 0, "EmbedDim must be divisible by NumHeads");
	static constexpr size_t MLPDim = 4 * EmbedDim;

public:
	TransformerBlock(size_t batchSize, unsigned seed = 42)
		: batchSize_(batchSize),
		  attn_(seed),
		  norm1_(), norm2_(),
		  kBuf_(batchSize * BlockSize * EmbedDim, Runtime::BufferMode::ReadWrite),
		  vBuf_(batchSize * BlockSize * EmbedDim, Runtime::BufferMode::ReadWrite),
		  attnBuf_(batchSize * BlockSize * EmbedDim, Runtime::BufferMode::ReadWrite),
		  mlpBuf_(batchSize * BlockSize * MLPDim, Runtime::BufferMode::ReadWrite) {
		std::vector<T> d1(MLPDim * EmbedDim);
		std::vector<T> d2(EmbedDim * MLPDim);

		float range = std::sqrt(6.0f / static_cast<float>(EmbedDim + MLPDim));
		unsigned s1 = seed + 10, s2 = seed + 11;

		for (size_t j = 0; j < MLPDim; j++)
			for (size_t i = 0; i < EmbedDim; i++) {
				s1 = s1 * 1664525u + 1013904223u;
				d1[j * EmbedDim + i] = (static_cast<float>(s1) / UINT32_MAX * 2.0f - 1.0f) * range;
			}
		for (size_t j = 0; j < EmbedDim; j++)
			for (size_t i = 0; i < MLPDim; i++) {
				s2 = s2 * 1664525u + 1013904223u;
				d2[j * MLPDim + i] = (static_cast<float>(s2) / UINT32_MAX * 2.0f - 1.0f) * range;
			}

		fc1W_ = Tensor<T, MLPDim, EmbedDim>(d1);
		fc2W_ = Tensor<T, EmbedDim, MLPDim>(d2);
	}

	void Setup() {
		attn_.Setup();
		norm1_.Setup();
		norm2_.Setup();

		fc1Ref_ = fc1W_.Bind();
		fc2Ref_ = fc2W_.Bind();
		fc1Ref_.ForEachParam([](auto &p) { AD::Param(p); });
		fc2Ref_.ForEachParam([](auto &p) { AD::Param(p); });

		kRef_    = kBuf_.Bind();
		vRef_    = vBuf_.Bind();
		attnRef_ = attnBuf_.Bind();
		mlpRef_  = mlpBuf_.Bind();
	}

	/**
	 * Forward pass for a single position.
	 *
	 * @param x        Residual stream buffer [batchSize * blockSize * embedDim]
	 * @param pos      Current position (0..blockSize-1)
	 * @param offset   Base offset for embed-dim buffers = batchIdx * blockSize * embedDim
	 */
	void Forward(const IR::Value::BufferRef<T> &x,
				 const IR::Value::Var<int> &pos,
				 const IR::Value::Expr<int> &offset) {
		constexpr int E = static_cast<int>(EmbedDim);
		constexpr int M = static_cast<int>(MLPDim);
		constexpr int BS = static_cast<int>(BlockSize);

		IR::Value::Expr<int> po = offset + pos * E;

		// MLP buffer: [batchSize * blockSize * MLPDim]
		// MLPDim = 4 * EmbedDim, so mlpOffset = 4 * offset, poM = 4 * po
		IR::Value::Expr<int> poM = po * MakeInt(static_cast<int>(MLPDim / EmbedDim));

		// --- RMSNorm → Attention → residual add ---
		norm1_.Forward(x, attnRef_, po);
		attn_.Forward(attnRef_, kRef_, vRef_, attnRef_, pos, offset);
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &d) {
			x[po + d] = x[po + d] + attnRef_[po + d];
		});

		// --- RMSNorm → MLP → residual add ---
		norm2_.Forward(x, attnRef_, po);

		// MLP fc1: expand E → 4E with ReLU
		GPU::Flow::For(MakeInt(0), MakeInt(M), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				sum = sum + fc1Ref_(o, i) * attnRef_[po + i];
			});
			mlpRef_[poM + o] = GPU::Math::Max(sum, MakeFloat(0.0f));
		});

		// MLP fc2: contract 4E → E, add to residual
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(M), [&](IR::Value::Var<int> &i) {
				sum = sum + fc2Ref_(o, i) * mlpRef_[poM + i];
			});
			x[po + o] = x[po + o] + sum;
		});
	}

	CausalSelfAttention<T, EmbedDim, NumHeads> &Attention() { return attn_; }
	Tensor<T, MLPDim, EmbedDim> &FC1() { return fc1W_; }
	Tensor<T, EmbedDim, MLPDim> &FC2() { return fc2W_; }
	static constexpr size_t ParamCount = CausalSelfAttention<T, EmbedDim, NumHeads>::TotalSize
		+ MLPDim * EmbedDim + EmbedDim * MLPDim;

private:
	size_t batchSize_;
	CausalSelfAttention<T, EmbedDim, NumHeads> attn_;
	RMSNorm<T, EmbedDim> norm1_, norm2_;

	Tensor<T, MLPDim, EmbedDim> fc1W_;
	Tensor<T, EmbedDim, MLPDim> fc2W_;
	TensorRef<T, MLPDim, EmbedDim> fc1Ref_;
	TensorRef<T, EmbedDim, MLPDim> fc2Ref_;

	Runtime::Buffer<T> kBuf_, vBuf_, attnBuf_, mlpBuf_;
	IR::Value::BufferRef<T> kRef_, vRef_, attnRef_, mlpRef_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_TRANSFORMER_H
