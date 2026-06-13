/**
 * @file TestAtomicAdvanced.cpp
 * @brief Advanced atomic operation tests: large-scale contention, min/max, bit ops,.
 */

#include <GPU.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;

#define TEST(name)                                                                                                     \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		try {

#define END_TEST                                                                                                       \
	std::cout << "PASSED\n";                                                                                           \
	}                                                                                                                  \
	catch (const std::exception &e) {                                                                                  \
		std::cout << "FAILED: " << e.what() << "\n";                                                                   \
		throw;                                                                                                         \
	}                                                                                                                  \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

#define ASSERT_EQ(a, b)                                                                                                \
	if ((a) != (b)) {                                                                                                  \
		throw std::runtime_error("Assertion failed: " #a " != " #b);                                                   \
	}

// =============================================================================
// Large-Scale Atomic Counter
// =============================================================================

TEST(atomic_counter_4096_threads)
constexpr int		 NUM_THREADS = 4096;
std::vector<int>	 zero(1, 0);
Runtime::Buffer<int> counter(zero);

Kernel1D			 kernel(
	[&](Int i) {
		auto cnt = counter.Bind();
		ExprBase::NotUse(AtomicAdd(cnt[0], MakeInt(1)));
	},
	256);

kernel.Dispatch(NUM_THREADS / 256, true);

std::vector<int> result(1);
counter.Download(result.data(), 1);
ASSERT_EQ(result[0], NUM_THREADS);
END_TEST

// =============================================================================
// Atomic Min / Max
// =============================================================================

TEST(atomic_min_max)
constexpr int		 N = 256;
std::vector<int>	 dataMin(1, 1000);
std::vector<int>	 dataMax(1, -1000);
Runtime::Buffer<int> bufMin(dataMin);
Runtime::Buffer<int> bufMax(dataMax);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto mn = bufMin.Bind();
		auto mx = bufMax.Bind();
		ExprBase::NotUse(AtomicMin(mn[0], id));
		ExprBase::NotUse(AtomicMax(mx[0], id));
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> outMin(1);
std::vector<int> outMax(1);
bufMin.Download(outMin.data(), 1);
bufMax.Download(outMax.data(), 1);
ASSERT_EQ(outMin[0], 0);   // min of 0..255
ASSERT_EQ(outMax[0], 255); // max of 0..255
END_TEST

// =============================================================================
// Atomic And / Or / Xor
// =============================================================================

TEST(atomic_and_or_xor)
constexpr int		 N = 256;
// Start with all bits set for AND test
std::vector<int>	 andData(1, 0xFFFFFFFF);
// Start with 0 for OR test
std::vector<int>	 orData(1, 0);
// Start with 0 for XOR test (even number of same XORs cancel)
std::vector<int>	 xorData(1, 0);

Runtime::Buffer<int> bufAnd(andData);
Runtime::Buffer<int> bufOr(orData);
Runtime::Buffer<int> bufXor(xorData);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto a = bufAnd.Bind();
		auto o = bufOr.Bind();
		auto x = bufXor.Bind();
		ExprBase::NotUse(AtomicAnd(a[0], id));
		ExprBase::NotUse(AtomicOr(o[0], id));
		ExprBase::NotUse(AtomicXor(x[0], id));
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> outAnd(1);
std::vector<int> outOr(1);
std::vector<int> outXor(1);
bufAnd.Download(outAnd.data(), 1);
bufOr.Download(outOr.data(), 1);
bufXor.Download(outXor.data(), 1);

// AND of 0..255: only bits common to all numbers remain.
// 0 contributes 0, so result is 0.
ASSERT_EQ(outAnd[0], 0);

// OR of 0..255: all bits that appear in any number.
// 0..255 = 0xFF
ASSERT_EQ(outOr[0], 255);

// XOR of 0..255:
// XOR(0..n) = [n, 1, n+1, 0] pattern based on n%4
// For n=255: XOR(0..255) = 255 (since 255%4==3 -> pattern gives n)
ASSERT_EQ(outXor[0], 0); // XOR(0..255) where 255%4==3 -> 0
END_TEST

// =============================================================================
// Atomic Compare-and-Swap
// =============================================================================

TEST(atomic_compare_swap)
constexpr int		 N = 256;
std::vector<int>	 data(1, 0);
Runtime::Buffer<int> buf(data);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		// Race to change 0 -> 1; only one thread should succeed in the CAS.
		ExprBase::NotUse(AtomicCompSwap(b[0], MakeInt(0), MakeInt(1)));
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> result(1);
buf.Download(result.data(), 1);
// At least one thread succeeded, so value should be 1
ASSERT_EQ(result[0], 1);
END_TEST

// =============================================================================
// Atomic Exchange
// =============================================================================

TEST(atomic_exchange)
constexpr int		 N = 256;
std::vector<int>	 data(1, -1);
Runtime::Buffer<int> buf(data);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		// Each thread tries to store its id; last one wins
		ExprBase::NotUse(AtomicExchange(b[0], id));
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> result(1);
buf.Download(result.data(), 1);
// Result should be in range [0, 255]
ASSERT(result[0] >= 0 && result[0] < 256);
END_TEST

// =============================================================================
// Atomic on Shared Memory
// =============================================================================

TEST(atomic_shared_memory)
constexpr int		 WG_SIZE = 256;
std::vector<int>	 data(1, 0);
Runtime::Buffer<int> buf(data);

Kernel1D			 kernel(
	[&](Int i) {
		SharedMemory<int, WG_SIZE> shared;
		Int						   lid = LocalThreadId();
		shared[lid]					   = MakeInt(0);
		Kernel1D::WorkgroupBarrier();

		// All threads atomically add to shared[0]
		ExprBase::NotUse(AtomicAdd(shared[0], MakeInt(1)));
		Kernel1D::WorkgroupBarrier();

		// Thread 0 writes the result to global buffer
		auto b = buf.Bind();
		If(lid == 0, [&]() { b[0] = shared[0]; });
	},
	WG_SIZE);

kernel.Dispatch(1, true);

std::vector<int> result(1);
buf.Download(result.data(), 1);
ASSERT_EQ(result[0], WG_SIZE);
END_TEST

// =============================================================================
// Atomic Sub
// =============================================================================

TEST(atomic_sub)
constexpr int		 N = 256;
std::vector<int>	 data(1, N);
Runtime::Buffer<int> buf(data);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		ExprBase::NotUse(AtomicSub(b[0], MakeInt(1)));
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> result(1);
buf.Download(result.data(), 1);
ASSERT_EQ(result[0], 0);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Atomic Advanced Tests         " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_atomic_counter_4096_threads();
		test_atomic_min_max();
		test_atomic_and_or_xor();
		test_atomic_compare_swap();
		test_atomic_exchange();
		test_atomic_shared_memory();
		test_atomic_sub();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All atomic advanced tests passed!     " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
