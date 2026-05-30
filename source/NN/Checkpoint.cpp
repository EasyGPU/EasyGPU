/**
 * @file Checkpoint.cpp
 * @brief Weight checkpoint I/O helper implementations.
 */

#include <NN/Checkpoint.h>

namespace GPU::NN::detail {

void WriteU32(FILE *f, uint32_t v) { std::fwrite(&v, sizeof(v), 1, f); }
void WriteU64(FILE *f, uint64_t v) { std::fwrite(&v, sizeof(v), 1, f); }
bool ReadU32(FILE *f, uint32_t &v) { return std::fread(&v, sizeof(v), 1, f) == 1; }
bool ReadU64(FILE *f, uint64_t &v) { return std::fread(&v, sizeof(v), 1, f) == 1; }

void WriteFloats(FILE *f, const float *data, size_t count) {
    std::fwrite(data, sizeof(float), count, f);
}

bool ReadFloats(FILE *f, float *data, size_t count) {
    return std::fread(data, sizeof(float), count, f) == count;
}

} // namespace GPU::NN::detail
