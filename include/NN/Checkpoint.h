#pragma once

/**
 * @file Checkpoint.h
 * @brief Weight checkpointing (save/load) for EasyGPU NN models.
 *
 * Usage:
 *   // Save all weights to a file
 *   SaveWeights("model.bin", fc1.Weight(), fc1.Bias(), fc2.Weight(), fc2.Bias());
 *
 *   // Load weights from a file
 *   LoadWeights("model.bin", fc1.Weight(), fc1.Bias(), fc2.Weight(), fc2.Bias());
 *
 * File format (binary): [numTensors: uint32] [size0: uint64] [data0...] [size1: uint64] [data1...] ...
 */

#ifndef EASYGPU_NN_CHECKPOINT_H
#define EASYGPU_NN_CHECKPOINT_H

#include <NN/Tensor.h>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::NN {

namespace detail {

inline void WriteU32(FILE *f, uint32_t v) { std::fwrite(&v, sizeof(v), 1, f); }
inline void WriteU64(FILE *f, uint64_t v) { std::fwrite(&v, sizeof(v), 1, f); }
inline bool ReadU32(FILE *f, uint32_t &v) { return std::fread(&v, sizeof(v), 1, f) == 1; }
inline bool ReadU64(FILE *f, uint64_t &v) { return std::fread(&v, sizeof(v), 1, f) == 1; }

inline void WriteFloats(FILE *f, const float *data, size_t count) {
	std::fwrite(data, sizeof(float), count, f);
}
inline bool ReadFloats(FILE *f, float *data, size_t count) {
	return std::fread(data, sizeof(float), count, f) == count;
}

} // namespace detail

/**
 * Save one or more tensors to a binary checkpoint file.
 *
 * @param path    File path (e.g., "checkpoint.bin")
 * @param tensors One or more Tensor<float, Dims...> references
 */
template <typename... Tensors>
void SaveWeights(const std::string &path, Tensors &... tensors) {
	FILE *f = std::fopen(path.c_str(), "wb");
	if (!f) throw std::runtime_error("SaveWeights: cannot open " + path + " for writing");

	uint32_t numTensors = static_cast<uint32_t>(sizeof...(Tensors));
	detail::WriteU32(f, numTensors);

	auto saveOne = [&](auto &tensor) {
		uint64_t count = static_cast<uint64_t>(tensor.Size());
		detail::WriteU64(f, count);
		detail::WriteFloats(f, tensor.Data(), count);
	};
	(saveOne(tensors), ...);

	std::fclose(f);
}

/**
 * Load one or more tensors from a binary checkpoint file.
 * Uploads data to GPU buffers after loading.
 *
 * @param path    File path
 * @param tensors One or more Tensor<float, Dims...> references (must match saved layout)
 */
template <typename... Tensors>
void LoadWeights(const std::string &path, Tensors &... tensors) {
	FILE *f = std::fopen(path.c_str(), "rb");
	if (!f) throw std::runtime_error("LoadWeights: cannot open " + path + " for reading");

	uint32_t numTensors = 0;
	if (!detail::ReadU32(f, numTensors))
		throw std::runtime_error("LoadWeights: failed to read tensor count");

	if (numTensors != static_cast<uint32_t>(sizeof...(Tensors)))
		throw std::runtime_error("LoadWeights: tensor count mismatch (file has " +
								 std::to_string(numTensors) + ", expected " +
								 std::to_string(sizeof...(Tensors)) + ")");

	auto loadOne = [&](auto &tensor) {
		uint64_t count = 0;
		if (!detail::ReadU64(f, count))
			throw std::runtime_error("LoadWeights: failed to read tensor size");
		if (count != static_cast<uint64_t>(tensor.Size()))
			throw std::runtime_error("LoadWeights: tensor size mismatch");
		if (!detail::ReadFloats(f, tensor.Data(), count))
			throw std::runtime_error("LoadWeights: failed to read tensor data");
		tensor.Upload();
	};
	(loadOne(tensors), ...);

	std::fclose(f);
}

} // namespace GPU::NN

#endif // EASYGPU_NN_CHECKPOINT_H
