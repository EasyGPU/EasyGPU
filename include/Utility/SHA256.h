#pragma once

/**
 * @file SHA256.h
 * @brief Cross-platform SHA-256 helper for stable cache keys.
 */

#include <string>
#include <string_view>

namespace GPU::Utility {

/** @brief Return the lowercase hexadecimal SHA-256 digest of the input bytes. */
std::string ComputeSHA256(std::string_view input);

} // namespace GPU::Utility
