#pragma once

/**
 * @file Embedding.h
 * @brief Token and positional embedding layers for EasyGPU NN.
 *
 *   TokenEmbedding<T, VocabSize, EmbedDim>      — row-wise gather from weight matrix
 *   PositionalEmbedding<T, BlockSize, EmbedDim> — position-indexed gather
 */

#ifndef EASYGPU_NN_EMBEDDING_H
#define EASYGPU_NN_EMBEDDING_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Utility/Helpers.h>

#include <cstddef>
#include <random>
#include <vector>

namespace GPU::NN {

// =============================================================================
// TokenEmbedding — learned embedding lookup by token ID
// =============================================================================

template <typename T, size_t VocabSize, size_t EmbedDim>
class TokenEmbedding {
	static_assert(std::is_same_v<T, float>, "TokenEmbedding only supports float");

public:
	TokenEmbedding(unsigned initSeed = 42) {
		std::vector<T> wData(VocabSize * EmbedDim);
		unsigned seed = initSeed;
		for (size_t v = 0; v < VocabSize; v++)
			for (size_t d = 0; d < EmbedDim; d++)
				wData[v * EmbedDim + d] = XavierInit(seed, VocabSize, EmbedDim);
		weight_ = Tensor<T, VocabSize, EmbedDim>(wData);
	}

	void Setup() {
		weightRef_ = weight_.Bind();
		weightRef_.ForEachParam([](auto &w) { AD::Param(w); });
	}

	/** Gather embedding vector at tokenId, write to out starting at outOffset. */
	void Forward(const IR::Value::Expr<int> &tokenId,
				 const IR::Value::BufferRef<T> &out,
				 const IR::Value::Expr<int> &outOffset) {
		constexpr int E = static_cast<int>(EmbedDim);
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &d) {
			out[outOffset + d] = weightRef_(tokenId, d);
		});
	}

	Tensor<T, VocabSize, EmbedDim> &Weight() { return weight_; }
	const Tensor<T, VocabSize, EmbedDim> &Weight() const { return weight_; }
	static constexpr size_t TotalSize = VocabSize * EmbedDim;

private:
	static float XavierInit(unsigned &seed, size_t fanIn, size_t fanOut) {
		float range = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
		seed = seed * 1664525u + 1013904223u;
		float r = static_cast<float>(static_cast<double>(seed) / UINT32_MAX);
		return (r * 2.0f - 1.0f) * range;
	}

	Tensor<T, VocabSize, EmbedDim> weight_;
	TensorRef<T, VocabSize, EmbedDim> weightRef_;
};

// =============================================================================
// PositionalEmbedding — learned position embedding lookup
// =============================================================================

template <typename T, size_t BlockSize, size_t EmbedDim>
class PositionalEmbedding {
	static_assert(std::is_same_v<T, float>, "PositionalEmbedding only supports float");

public:
	PositionalEmbedding(unsigned initSeed = 123) {
		std::vector<T> wData(BlockSize * EmbedDim);
		unsigned seed = initSeed;
		for (size_t p = 0; p < BlockSize; p++)
			for (size_t d = 0; d < EmbedDim; d++)
				wData[p * EmbedDim + d] = XavierInit(seed, BlockSize, EmbedDim);
		weight_ = Tensor<T, BlockSize, EmbedDim>(wData);
	}

	void Setup() {
		weightRef_ = weight_.Bind();
		weightRef_.ForEachParam([](auto &w) { AD::Param(w); });
	}

	void Forward(const IR::Value::Expr<int> &pos,
				 const IR::Value::BufferRef<T> &out,
				 const IR::Value::Expr<int> &outOffset) {
		constexpr int E = static_cast<int>(EmbedDim);
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &d) {
			out[outOffset + d] = out[outOffset + d] + weightRef_(pos, d);
		});
	}

	Tensor<T, BlockSize, EmbedDim> &Weight() { return weight_; }
	const Tensor<T, BlockSize, EmbedDim> &Weight() const { return weight_; }
	static constexpr size_t TotalSize = BlockSize * EmbedDim;

private:
	static float XavierInit(unsigned &seed, size_t fanIn, size_t fanOut) {
		float range = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
		seed = seed * 1664525u + 1013904223u;
		float r = static_cast<float>(static_cast<double>(seed) / UINT32_MAX);
		return (r * 2.0f - 1.0f) * range;
	}

	Tensor<T, BlockSize, EmbedDim> weight_;
	TensorRef<T, BlockSize, EmbedDim> weightRef_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_EMBEDDING_H
