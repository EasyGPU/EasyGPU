/**
 * @file SHA256.cpp
 * @brief Cross-platform SHA-256 helper implementation.
 */

#include <Utility/SHA256.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <wincrypt.h>
#elif defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/evp.h>
#endif

namespace GPU::Utility {

std::string ComputeSHA256(std::string_view input) {
	unsigned char hash[32] = {};
	bool hashComputed = false;

#ifdef _WIN32
	HCRYPTPROV hProv = 0;
	HCRYPTHASH hHash = 0;
	DWORD hashLen = sizeof(hash);

	if (input.size() <= static_cast<size_t>(std::numeric_limits<DWORD>::max()) &&
		CryptAcquireContext(&hProv, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
		if (CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) {
			hashComputed = CryptHashData(hHash, reinterpret_cast<const BYTE *>(input.data()),
										static_cast<DWORD>(input.size()), 0) &&
						   CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0);
			CryptDestroyHash(hHash);
		}
		CryptReleaseContext(hProv, 0);
	}
#elif defined(__APPLE__)
	if (input.size() <= static_cast<size_t>(std::numeric_limits<CC_LONG>::max())) {
		CC_SHA256_CTX ctx;
		CC_SHA256_Init(&ctx);
		CC_SHA256_Update(&ctx, input.data(), static_cast<CC_LONG>(input.size()));
		CC_SHA256_Final(hash, &ctx);
		hashComputed = true;
	}
#else
	EVP_MD_CTX *ctx = EVP_MD_CTX_new();
	if (ctx != nullptr) {
		hashComputed = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
					   EVP_DigestUpdate(ctx, input.data(), input.size()) == 1 &&
					   EVP_DigestFinal_ex(ctx, hash, nullptr) == 1;
		EVP_MD_CTX_free(ctx);
	}
#endif

	if (!hashComputed) {
		throw std::runtime_error("SHA256 hash computation failed");
	}

	std::ostringstream result;
	for (unsigned char byte : hash) {
		result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
	}
	return result.str();
}

} // namespace GPU::Utility
