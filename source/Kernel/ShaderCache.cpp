/**
 * @file ShaderCache.cpp
 * @brief Implementation of in-memory shader binary cache.
 */

#include <Kernel/ShaderCache.h>
#include <Utility/SHA256.h>

#include <chrono>

namespace GPU::Kernel {

// =============================================================================
// ShaderCache Implementation
// =============================================================================

ShaderCache::ShaderCache() = default;

std::optional<CacheEntry> ShaderCache::Lookup(const std::string &shaderHash, uint32_t backendType) const {
	std::lock_guard<std::mutex> lock(_mutex);

	std::string					key = std::to_string(backendType) + ":" + shaderHash;
	auto						it	= _entries.find(key);
	if (it != _entries.end()) {
		return it->second;
	}
	return std::nullopt;
}

bool ShaderCache::Store(const std::string &shaderHash, uint32_t backendType, uint32_t format,
						const std::vector<uint8_t> &data) {
	std::lock_guard<std::mutex> lock(_mutex);

	CacheEntry					entry;
	entry.timestamp =
		std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	entry.dataSize	  = static_cast<uint32_t>(data.size());
	entry.backendType = backendType;
	entry.format	  = format;
	entry.shaderHash  = shaderHash;
	entry.data		  = data;

	std::string key	  = std::to_string(backendType) + ":" + shaderHash;
	_entries[key]	  = std::move(entry);

	return true;
}

void ShaderCache::GetStats(size_t &totalEntries, size_t &totalBytes) const {
	std::lock_guard<std::mutex> lock(_mutex);

	totalEntries = _entries.size();
	totalBytes	 = 0;
	for (const auto &[key, entry] : _entries) {
		totalBytes += entry.data.size();
	}
}

void ShaderCache::Clear() {
	std::lock_guard<std::mutex> lock(_mutex);
	_entries.clear();
}

std::string ShaderCache::ComputeShaderHash(const std::string &sourceCode) {
	return Utility::ComputeSHA256(sourceCode);
}

// =============================================================================
// GlobalShaderCache Implementation
// =============================================================================

std::unique_ptr<ShaderCache> GlobalShaderCache::_cache;
std::mutex					 GlobalShaderCache::_mutex;

ShaderCache					&GlobalShaderCache::Get() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_cache) {
		_cache = std::make_unique<ShaderCache>();
	}

	return *_cache;
}

void GlobalShaderCache::Clear() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_cache) {
		_cache->Clear();
	}
}

bool GlobalShaderCache::IsEnabled() {
	std::lock_guard<std::mutex> lock(_mutex);
	return _cache != nullptr;
}

} // namespace GPU::Kernel
