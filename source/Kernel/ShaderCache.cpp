/**
 * @file ShaderCache.cpp
 * @brief Implementation of in-memory shader binary cache
 */

#include <Kernel/ShaderCache.h>

#include <chrono>
#include <iomanip>
#include <sstream>

// For SHA256 hash computation - use standard library where possible
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincrypt.h>
#else
#include <openssl/evp.h>
#endif

namespace GPU::Kernel {

// Simple SHA256 implementation wrapper
class SHA256 {
public:
	static std::string Hash(const std::string &input) {
		unsigned char hash[32];

#ifdef _WIN32
		// Windows Cryptography API
		HCRYPTPROV hProv   = 0;
		HCRYPTHASH hHash   = 0;
		DWORD	   hashLen = 32;

		if (CryptAcquireContext(&hProv, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
			if (CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) {
				CryptHashData(hHash, reinterpret_cast<const BYTE *>(input.c_str()), static_cast<DWORD>(input.size()),
							  0);
				CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0);
				CryptDestroyHash(hHash);
			}
			CryptReleaseContext(hProv, 0);
		}
#else
		// OpenSSL EVP API
		EVP_MD_CTX *ctx = EVP_MD_CTX_new();
		if (ctx) {
			EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
			EVP_DigestUpdate(ctx, input.c_str(), input.size());
			EVP_DigestFinal_ex(ctx, hash, nullptr);
			EVP_MD_CTX_free(ctx);
		}
#endif

		// Convert to hex string
		std::ostringstream oss;
		for (int i = 0; i < 32; ++i) {
			oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
		}
		return oss.str();
	}
};

// =============================================================================
// ShaderCache Implementation
// =============================================================================

ShaderCache::ShaderCache() = default;

const CacheEntry *ShaderCache::Lookup(const std::string &shaderHash, uint32_t backendType) const {
	std::lock_guard<std::mutex> lock(_mutex);

	std::string					key = std::to_string(backendType) + ":" + shaderHash;
	auto						it	= _entries.find(key);
	if (it != _entries.end()) {
		return &it->second;
	}
	return nullptr;
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
	return SHA256::Hash(sourceCode);
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
