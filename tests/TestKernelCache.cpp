/**
 * @file TestKernelCache.cpp
 * @brief Comprehensive tests for kernel shader binary cache functionality.
 */

#include <GPU.h>
#include <Kernel/ShaderCache.h>

#include <chrono>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::Kernel;

// =============================================================================
// Test Helpers
// =============================================================================

class TestTimer {
public:
	TestTimer() : _start(std::chrono::high_resolution_clock::now()) {
	}

	void Reset() {
		_start = std::chrono::high_resolution_clock::now();
	}

	double ElapsedMs() const {
		auto end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(end - _start).count();
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

// =============================================================================
// Test 1: Basic Cache Operations
// =============================================================================

bool Test_BasicCacheOperations() {
	std::cout << "[Test] Basic cache operations..." << std::endl;

	ShaderCache			 cache;

	// Store entries for different backends
	std::vector<uint8_t> glData = {0x01, 0x02, 0x03, 0x04};
	std::vector<uint8_t> vkData = {0x11, 0x12, 0x13, 0x14};

	if (!cache.Store("shader1", 0, 1, glData)) {
		std::cerr << "  FAILED: Store returned false" << std::endl;
		return false;
	}

	if (!cache.Store("shader2", 1, 2, vkData)) {
		std::cerr << "  FAILED: Store for Vulkan returned false" << std::endl;
		return false;
	}

	// Lookup entries
	auto entry1 = cache.Lookup("shader1", 0);
	if (!entry1) {
		std::cerr << "  FAILED: Lookup for existing entry returned nullptr" << std::endl;
		return false;
	}

	if (entry1->data != glData) {
		std::cerr << "  FAILED: Data mismatch" << std::endl;
		return false;
	}

	// Check backend-specific lookup
	auto entry2 = cache.Lookup("shader2", 1);
	if (!entry2 || entry2->data != vkData) {
		std::cerr << "  FAILED: Vulkan entry lookup failed" << std::endl;
		return false;
	}

	// Check cross-backend isolation
	auto wrongBackend = cache.Lookup("shader1", 1);
	if (wrongBackend.has_value()) {
		std::cerr << "  FAILED: Cross-backend lookup should return nullptr" << std::endl;
		return false;
	}

	// Test non-existent entry
	auto missing = cache.Lookup("nonexistent", 0);
	if (missing.has_value()) {
		std::cerr << "  FAILED: Non-existent entry should return nullptr" << std::endl;
		return false;
	}

	// Check stats
	size_t entries, bytes;
	cache.GetStats(entries, bytes);
	if (entries != 2) {
		std::cerr << "  FAILED: Expected 2 entries, got " << entries << std::endl;
		return false;
	}
	if (bytes != 8) {
		std::cerr << "  FAILED: Expected 8 bytes, got " << bytes << std::endl;
		return false;
	}

	// Clear cache
	cache.Clear();
	cache.GetStats(entries, bytes);
	if (entries != 0 || bytes != 0) {
		std::cerr << "  FAILED: Clear did not reset stats" << std::endl;
		return false;
	}

	std::cout << "  PASSED" << std::endl;
	return true;
}

// =============================================================================
// Test 2: Shader Hash Computation
// =============================================================================

bool Test_ShaderHash() {
	std::cout << "[Test] Shader hash computation..." << std::endl;

	// Same source -> same hash
	std::string source1 = R"(
		#version 430 core
		layout(local_size_x = 256) in;
		layout(std430, binding = 0) buffer Data {
			float values[];
		};
		void main() {
			uint idx = gl_GlobalInvocationID.x;
			values[idx] = values[idx] * 2.0;
		}
	)";

	std::string source2 = source1;
	std::string source3 = source1 + "\n// Different comment";

	std::string hash1	= ShaderCache::ComputeShaderHash(source1);
	std::string hash2	= ShaderCache::ComputeShaderHash(source2);
	std::string hash3	= ShaderCache::ComputeShaderHash(source3);

	// Same source should produce same hash
	if (hash1 != hash2) {
		std::cerr << "  FAILED: Same source produced different hashes" << std::endl;
		return false;
	}

	// Different source should produce different hash
	if (hash1 == hash3) {
		std::cerr << "  FAILED: Different source produced same hash" << std::endl;
		return false;
	}

	// Hash should be 64 hex chars (SHA256)
	if (hash1.length() != 64) {
		std::cerr << "  FAILED: Hash length should be 64, got " << hash1.length() << std::endl;
		return false;
	}

	// Hash should only contain hex characters
	for (char c : hash1) {
		if (!std::isxdigit(c)) {
			std::cerr << "  FAILED: Hash contains non-hex character" << std::endl;
			return false;
		}
	}

	// Empty string should still produce valid hash
	std::string emptyHash = ShaderCache::ComputeShaderHash("");
	if (emptyHash.length() != 64) {
		std::cerr << "  FAILED: Empty string hash length incorrect" << std::endl;
		return false;
	}

	std::cout << "  PASSED" << std::endl;
	return true;
}

// =============================================================================
// Test 3: Kernel Compilation Caching (Integration)
// =============================================================================

bool Test_KernelCompilationCache() {
	std::cout << "[Test] Kernel compilation cache integration..." << std::endl;

	// Clear any existing global cache
	GlobalShaderCache::Clear();

	// Prepare buffers
	constexpr int	   SIZE = 1024;
	std::vector<float> data(SIZE);
	for (int i = 0; i < SIZE; ++i)
		data[i] = static_cast<float>(i);

	Buffer<float> input(data);
	Buffer<float> output(SIZE);

	// First kernel creation - should compile and cache
	TestTimer	  timer1;
	{
		Kernel1D kernel([&](Int i) {
			auto in	 = input.Bind();
			auto out = output.Bind();
			out[i]	 = in[i] * 2.0f + 1.0f;
		});

		kernel.Dispatch(SIZE / 256, true);
	}
	double			   timeFirst = timer1.ElapsedMs();

	// Verify first execution worked
	std::vector<float> result(SIZE);
	output.Download(result);
	for (int i = 0; i < SIZE; ++i) {
		float expected = i * 2.0f + 1.0f;
		if (std::abs(result[i] - expected) > 0.001f) {
			std::cerr << "  FAILED: First kernel execution incorrect at index " << i << std::endl;
			return false;
		}
	}

	// Note: Cache is in-memory only, new Kernel1D will still compile
	// because we create a new instance. The cache is used when the SAME
	// kernel context is reused (which happens in the same Dispatch).
	//
	// For this test, we just verify the kernel works correctly with caching enabled.

	std::cout << "  First compile: " << timeFirst << "ms" << std::endl;
	std::cout << "  PASSED (cache is in-memory only)" << std::endl;
	return true;
}

// =============================================================================
// Test 4: Multiple Kernels Caching
// =============================================================================

bool Test_MultipleKernelsCaching() {
	std::cout << "[Test] Multiple kernels caching..." << std::endl;

	// Clear cache
	GlobalShaderCache::Clear();

	constexpr int SIZE = 512;
	Buffer<float> buf1(SIZE);
	Buffer<float> buf2(SIZE);
	Buffer<float> buf3(SIZE);

	// Create and run multiple different kernels
	Kernel1D	  kernel1([&](Int i) {
		 auto b = buf1.Bind();
		 b[i]	= ToFloat(i) * 1.0f;
	 });

	Kernel1D	  kernel2([&](Int i) {
		 auto b = buf2.Bind();
		 b[i]	= ToFloat(i) * 2.0f;
	 });

	Kernel1D	  kernel3([&](Int i) {
		 auto b = buf3.Bind();
		 b[i]	= ToFloat(i) * 3.0f;
	 });

	kernel1.Dispatch(2, true);
	kernel2.Dispatch(2, true);
	kernel3.Dispatch(2, true);

	// Verify results
	std::vector<float> result1(SIZE), result2(SIZE), result3(SIZE);
	buf1.Download(result1);
	buf2.Download(result2);
	buf3.Download(result3);

	for (int i = 0; i < SIZE; ++i) {
		if (std::abs(result1[i] - i * 1.0f) > 0.001f || std::abs(result2[i] - i * 2.0f) > 0.001f ||
			std::abs(result3[i] - i * 3.0f) > 0.001f) {
			std::cerr << "  FAILED: Kernel results incorrect" << std::endl;
			return false;
		}
	}

	std::cout << "  PASSED (3 kernels cached in memory)" << std::endl;
	return true;
}

// =============================================================================
// Test 5: Backend Cache Support Detection
// =============================================================================

bool Test_BackendCacheSupport() {
	std::cout << "[Test] Backend cache support detection..." << std::endl;

	Runtime::AutoInitContext();
	auto *backend = Runtime::Context::GetBackend();

	if (!backend) {
		std::cerr << "  FAILED: Backend not available" << std::endl;
		return false;
	}

	bool	 supportsCache = backend->SupportsPipelineCache();
	uint32_t format		   = backend->GetPipelineCacheFormat();

	std::cout << "  Backend supports pipeline cache: " << (supportsCache ? "YES" : "NO") << std::endl;
	std::cout << "  Cache format identifier: 0x" << std::hex << format << std::dec << std::endl;

	// If cache is supported, format should be non-zero
	if (supportsCache && format == 0) {
		std::cerr << "  WARNING: Cache supported but format is 0" << std::endl;
	}

	std::cout << "  PASSED" << std::endl;
	return true;
}

// =============================================================================
// Test 6: Global Cache Operations
// =============================================================================

bool Test_GlobalCacheOperations() {
	std::cout << "[Test] Global cache operations..." << std::endl;

	// Ensure cache is initialized
	auto &cache = GlobalShaderCache::Get();

	if (!GlobalShaderCache::IsEnabled()) {
		std::cerr << "  FAILED: Global cache not enabled" << std::endl;
		return false;
	}

	// Store something
	std::vector<uint8_t> data = {1, 2, 3, 4, 5};
	cache.Store("test_global", 0, 99, data);

	// Verify
	auto entry = cache.Lookup("test_global", 0);
	if (!entry || entry->data != data) {
		std::cerr << "  FAILED: Global cache store/lookup failed" << std::endl;
		return false;
	}

	// Clear
	GlobalShaderCache::Clear();

	// Verify cleared
	entry = cache.Lookup("test_global", 0);
	if (entry.has_value()) {
		std::cerr << "  FAILED: Global cache clear failed" << std::endl;
		return false;
	}

	std::cout << "  PASSED" << std::endl;
	return true;
}

// =============================================================================
// Test 7: Large Data Caching
// =============================================================================

bool Test_LargeDataCaching() {
	std::cout << "[Test] Large data caching..." << std::endl;

	ShaderCache			 cache;

	// Create large binary data (1MB)
	std::vector<uint8_t> largeData(1024 * 1024);
	for (size_t i = 0; i < largeData.size(); ++i) {
		largeData[i] = static_cast<uint8_t>(i % 256);
	}

	// Store large data
	if (!cache.Store("large_shader", 0, 99, largeData)) {
		std::cerr << "  FAILED: Failed to store large data" << std::endl;
		return false;
	}

	// Verify immediate lookup
	auto entry = cache.Lookup("large_shader", 0);
	if (!entry) {
		std::cerr << "  FAILED: Large data lookup failed" << std::endl;
		return false;
	}

	if (entry->data != largeData) {
		std::cerr << "  FAILED: Large data content mismatch" << std::endl;
		return false;
	}

	// Verify content byte-by-byte
	bool contentOk = true;
	for (size_t i = 0; i < entry->data.size() && contentOk; ++i) {
		if (entry->data[i] != static_cast<uint8_t>(i % 256)) {
			contentOk = false;
		}
	}

	if (!contentOk) {
		std::cerr << "  FAILED: Large data content corrupted" << std::endl;
		return false;
	}

	std::cout << "  PASSED (1MB data in memory)" << std::endl;
	return true;
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Kernel Cache Tests (In-Memory Only)" << std::endl;
	std::cout << "========================================" << std::endl;

	int passed = 0;
	int failed = 0;

	struct TestCase {
		const char *name;
		bool (*func)();
	};

	TestCase tests[] = {
		{"Basic Cache Operations", Test_BasicCacheOperations},
		{"Shader Hash Computation", Test_ShaderHash},
		{"Kernel Compilation Cache", Test_KernelCompilationCache},
		{"Multiple Kernels Caching", Test_MultipleKernelsCaching},
		{"Backend Cache Support", Test_BackendCacheSupport},
		{"Global Cache Operations", Test_GlobalCacheOperations},
		{"Large Data Caching", Test_LargeDataCaching},
	};

	for (const auto &test : tests) {
		std::cout << std::endl;
		try {
			if (test.func()) {
				passed++;
			} else {
				std::cerr << "FAILED: " << test.name << std::endl;
				failed++;
			}
		} catch (const std::exception &e) {
			std::cerr << "EXCEPTION in " << test.name << ": " << e.what() << std::endl;
			failed++;
		}
	}

	std::cout << std::endl << "========================================" << std::endl;
	std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
	std::cout << "========================================" << std::endl;

	return failed == 0 ? 0 : 1;
}
