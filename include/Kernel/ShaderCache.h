#pragma once

/**
 * @file ShaderCache.h
 * @brief In-memory shader binary cache for kernel compilation acceleration
 * 
 * This module provides runtime caching of compiled GPU programs to avoid
 * recompilation overhead within a single application session. Cache is
 * NOT persisted to disk - it only exists in memory.
 */

#ifndef EASYGPU_SHADERCACHE_H
#define EASYGPU_SHADERCACHE_H

#include <Backend/Backend.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GPU::Kernel {

/**
 * Cache entry metadata for a single cached shader binary
 */
struct CacheEntry {
	uint64_t	timestamp;      // When this entry was created
	uint32_t	dataSize;      // Size of binary data in bytes
	uint32_t	backendType;   // Backend type identifier (OpenGL/Vulkan)
	uint32_t	format;        // Backend-specific format identifier
	std::string shaderHash;  // Source code hash for verification
	std::vector<uint8_t> data; // Binary data
};

/**
 * In-memory shader cache manager
 * 
 * This class manages an in-memory cache of compiled shader binaries.
 * It is thread-safe but NOT persisted to disk.
 */
class ShaderCache {
public:
	/**
	 * Create an in-memory shader cache
	 */
	ShaderCache();
	
	~ShaderCache() = default;

	// Non-copyable, movable
	ShaderCache(const ShaderCache&) = delete;
	ShaderCache& operator=(const ShaderCache&) = delete;
	ShaderCache(ShaderCache&&) noexcept = default;
	ShaderCache& operator=(ShaderCache&&) noexcept = default;

	/**
	 * Look up a cached binary by shader source hash
	 * @param shaderHash Hash of the shader source code
	 * @param backendType Required backend type
	 * @return Pointer to entry if found, nullptr otherwise
	 */
	const CacheEntry* Lookup(const std::string& shaderHash, uint32_t backendType) const;

	/**
	 * Store a binary in the cache
	 * @param shaderHash Hash of the shader source code
	 * @param backendType Backend type identifier
	 * @param format Backend-specific format code
	 * @param data Binary data to store
	 * @return True if stored successfully
	 */
	bool Store(const std::string& shaderHash, uint32_t backendType, uint32_t format, 
	           const std::vector<uint8_t>& data);

	/**
	 * Get cache statistics
	 * @param[out] totalEntries Number of entries in cache
	 * @param[out] totalBytes Total size of cached data
	 */
	void GetStats(size_t& totalEntries, size_t& totalBytes) const;

	/**
	 * Clear all cached entries
	 */
	void Clear();

	/**
	 * Generate a hash key from shader source code
	 * @param sourceCode GLSL source code
	 * @return Hash string suitable for cache lookup
	 */
	static std::string ComputeShaderHash(const std::string& sourceCode);

private:
	// Key: "backendType:shaderHash"
	std::unordered_map<std::string, CacheEntry> _entries;
	
	mutable std::mutex _mutex;
};

/**
 * Global shader cache instance for automatic kernel caching
 * 
 * This singleton provides a default cache that kernels will use
 * when caching is enabled. Cache is in-memory only.
 */
class GlobalShaderCache {
public:
	/**
	 * Get the global shader cache instance
	 */
	static ShaderCache& Get();

	/**
	 * Clear the global cache
	 */
	static void Clear();

	/**
	 * Check if global cache is initialized
	 */
	static bool IsEnabled();

private:
	GlobalShaderCache() = default;
	
	static std::unique_ptr<ShaderCache> _cache;
	static std::mutex _mutex;
};

} // namespace GPU::Kernel

#endif // EASYGPU_SHADERCACHE_H
