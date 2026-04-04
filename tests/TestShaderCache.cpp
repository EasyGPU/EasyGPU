/**
 * @file TestShaderCache.cpp
 * @brief Tests for in-memory shader binary cache functionality
 */

#include <GPU.h>
#include <Kernel/ShaderCache.h>

#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::Kernel;

/**
 * Test basic cache operations
 */
bool TestBasicCacheOperations() {
	std::cout << "Testing basic cache operations..." << std::endl;

	ShaderCache			 cache;

	// Store an entry
	std::vector<uint8_t> testData = {0x01, 0x02, 0x03, 0x04, 0x05};
	bool				 stored	  = cache.Store("test_hash_123", 0, 1, testData);
	if (!stored) {
		std::cerr << "Failed to store cache entry" << std::endl;
		return false;
	}

	// Lookup the entry
	const CacheEntry *entry = cache.Lookup("test_hash_123", 0);
	if (!entry) {
		std::cerr << "Failed to lookup cache entry" << std::endl;
		return false;
	}

	if (entry->data != testData) {
		std::cerr << "Cache data mismatch" << std::endl;
		return false;
	}

	// Lookup non-existent entry
	const CacheEntry *missing = cache.Lookup("non_existent", 0);
	if (missing != nullptr) {
		std::cerr << "Should not find non-existent entry" << std::endl;
		return false;
	}

	// Check stats
	size_t entries, bytes;
	cache.GetStats(entries, bytes);
	if (entries != 1 || bytes != 5) {
		std::cerr << "Cache stats incorrect" << std::endl;
		return false;
	}

	// Clear cache
	cache.Clear();
	cache.GetStats(entries, bytes);
	if (entries != 0 || bytes != 0) {
		std::cerr << "Clear did not reset stats" << std::endl;
		return false;
	}

	std::cout << "  Basic cache operations: PASSED" << std::endl;
	return true;
}

/**
 * Test shader hash computation
 */
bool TestShaderHash() {
	std::cout << "Testing shader hash computation..." << std::endl;

	std::string source1 = "void main() { int x = 42; }";
	std::string source2 = "void main() { int x = 42; }";
	std::string source3 = "void main() { int x = 43; }";

	std::string hash1	= ShaderCache::ComputeShaderHash(source1);
	std::string hash2	= ShaderCache::ComputeShaderHash(source2);
	std::string hash3	= ShaderCache::ComputeShaderHash(source3);

	// Same source should produce same hash
	if (hash1 != hash2) {
		std::cerr << "Same source produced different hashes" << std::endl;
		return false;
	}

	// Different source should produce different hash
	if (hash1 == hash3) {
		std::cerr << "Different source produced same hash" << std::endl;
		return false;
	}

	// Hash should be 64 characters (SHA256 hex)
	if (hash1.length() != 64) {
		std::cerr << "Hash length should be 64, got " << hash1.length() << std::endl;
		return false;
	}

	std::cout << "  Shader hash computation: PASSED" << std::endl;
	return true;
}

/**
 * Test cache statistics
 */
bool TestCacheStats() {
	std::cout << "Testing cache statistics..." << std::endl;

	ShaderCache cache;

	size_t		entries, bytes;
	cache.GetStats(entries, bytes);

	if (entries != 0 || bytes != 0) {
		std::cerr << "Empty cache should have zero stats" << std::endl;
		return false;
	}

	// Add some entries
	std::vector<uint8_t> data1(100, 0xAB);
	std::vector<uint8_t> data2(200, 0xCD);

	cache.Store("hash1", 0, 1, data1);
	cache.Store("hash2", 0, 1, data2);

	cache.GetStats(entries, bytes);

	if (entries != 2) {
		std::cerr << "Expected 2 entries, got " << entries << std::endl;
		return false;
	}

	if (bytes != 300) {
		std::cerr << "Expected 300 bytes, got " << bytes << std::endl;
		return false;
	}

	std::cout << "  Cache statistics: PASSED" << std::endl;
	return true;
}

/**
 * Test global shader cache
 */
bool TestGlobalShaderCache() {
	std::cout << "Testing global shader cache..." << std::endl;

	// Clear any existing cache
	GlobalShaderCache::Clear();

	// Get the global cache (this will initialize it)
	auto				&cache = GlobalShaderCache::Get();

	// Store something
	std::vector<uint8_t> data  = {1, 2, 3, 4, 5};
	cache.Store("global_test", 0, 99, data);

	// Verify
	const CacheEntry *entry = cache.Lookup("global_test", 0);
	if (!entry || entry->data != data) {
		std::cerr << "Global cache store/lookup failed" << std::endl;
		return false;
	}

	std::cout << "  Global shader cache: PASSED" << std::endl;
	return true;
}

/**
 * Main entry point
 */
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Shader Cache Tests (In-Memory Only)" << std::endl;
	std::cout << "========================================" << std::endl;

	int	 passed	 = 0;
	int	 failed	 = 0;

	auto runTest = [&](const char *name, bool (*testFunc)()) {
		std::cout << std::endl << "Running: " << name << std::endl;
		try {
			if (testFunc()) {
				passed++;
			} else {
				std::cerr << "FAILED: " << name << std::endl;
				failed++;
			}
		} catch (const std::exception &e) {
			std::cerr << "EXCEPTION in " << name << ": " << e.what() << std::endl;
			failed++;
		}
	};

	runTest("BasicCacheOperations", TestBasicCacheOperations);
	runTest("ShaderHash", TestShaderHash);
	runTest("CacheStats", TestCacheStats);
	runTest("GlobalShaderCache", TestGlobalShaderCache);

	std::cout << std::endl << "========================================" << std::endl;
	std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
	std::cout << "========================================" << std::endl;

	return failed == 0 ? 0 : 1;
}
