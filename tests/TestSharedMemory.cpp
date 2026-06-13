/**
 * @file TestSharedMemory.cpp
 * @brief Test for SharedMemory, Atomic Operations, and Parallel Primitives.
 */

#include <GPU.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace GPU;

// Test 1: Basic SharedMemory declaration and usage
void test_shared_memory_basic() {
	std::cout << "Test: SharedMemory Basic Declaration..." << std::endl;

	// Use InspectorKernel to check generated code
	InspectorKernel1D inspector([](Int i) {
		// Declare shared memory array
		SharedMemory<float, 256> shared;

		// Get local thread ID
		Int						 localId = LocalThreadId();

		// Store value to shared memory
		shared[localId]					 = MakeFloat(1.0f);

		// Barrier to synchronize
		Kernel1D::WorkgroupBarrier();

		// Read from shared memory (from another thread)
		Var<float> val = shared[(localId + 1) % 256];
	});

	std::string		  code = inspector.GetCode();

	// Check that shared memory declaration exists
	if (code.find("shared float") == std::string::npos) {
		std::cerr << "FAIL: Shared memory declaration not found in generated code" << std::endl;
		std::cout << "Generated code:\n" << code << std::endl;
		return;
	}

	// Check that barrier is called
	if (code.find("barrier()") == std::string::npos) {
		std::cerr << "FAIL: barrier() call not found in generated code" << std::endl;
		return;
	}

	std::cout << "PASS: SharedMemory basic declaration works" << std::endl;
}

// Test 2: Atomic Operations
void test_atomic_operations() {
	std::cout << "Test: Atomic Operations..." << std::endl;

	InspectorKernel1D inspector([](Int i) {
		// Buffer for atomic operations
		Buffer<int> counter(1);

		auto		buf = counter.Bind();

		// Various atomic operations
		ExprBase::NotUse(AtomicAdd(buf[0], MakeInt(1)));
		ExprBase::NotUse(AtomicSub(buf[0], MakeInt(1)));
		ExprBase::NotUse(AtomicMin(buf[0], MakeInt(100)));
		ExprBase::NotUse(AtomicMax(buf[0], MakeInt(0)));
		ExprBase::NotUse(AtomicAnd(buf[0], MakeInt(0xFF)));
		ExprBase::NotUse(AtomicOr(buf[0], MakeInt(0x100)));
		ExprBase::NotUse(AtomicXor(buf[0], MakeInt(0x1)));
		ExprBase::NotUse(AtomicExchange(buf[0], MakeInt(42)));
		ExprBase::NotUse(AtomicCompSwap(buf[0], MakeInt(42), MakeInt(100)));
	});

	std::string		  code				= inspector.GetCode();

	// Check for atomic function calls
	bool			  hasAtomicAdd		= code.find("atomicAdd") != std::string::npos;
	bool			  hasAtomicMin		= code.find("atomicMin") != std::string::npos;
	bool			  hasAtomicMax		= code.find("atomicMax") != std::string::npos;
	bool			  hasAtomicAnd		= code.find("atomicAnd") != std::string::npos;
	bool			  hasAtomicOr		= code.find("atomicOr") != std::string::npos;
	bool			  hasAtomicXor		= code.find("atomicXor") != std::string::npos;
	bool			  hasAtomicExchange = code.find("atomicExchange") != std::string::npos;
	bool			  hasAtomicCompSwap = code.find("atomicCompSwap") != std::string::npos;

	if (!hasAtomicAdd || !hasAtomicMin || !hasAtomicMax || !hasAtomicAnd || !hasAtomicOr || !hasAtomicXor ||
		!hasAtomicExchange || !hasAtomicCompSwap) {
		std::cerr << "FAIL: Some atomic operations not found in generated code" << std::endl;
		std::cout << "Generated code:\n" << code << std::endl;
		return;
	}

	std::cout << "PASS: All atomic operations generate correct code" << std::endl;
}

// Test 3: Atomic with SharedMemory
void test_atomic_shared_memory() {
	std::cout << "Test: Atomic Operations with SharedMemory..." << std::endl;

	InspectorKernel1D inspector([](Int i) {
		SharedMemory<int, 256> shared;
		Int					   localId = LocalThreadId();

		// Initialize
		shared[localId]				   = MakeInt(0);
		Kernel1D::WorkgroupBarrier();

		// Atomic add to shared memory
		ExprBase::NotUse(AtomicAdd(shared[localId], MakeInt(1)));
	});

	std::string		  code = inspector.GetCode();

	// Check for atomicAdd with shared memory
	if (code.find("atomicAdd") == std::string::npos) {
		std::cerr << "FAIL: atomicAdd not found" << std::endl;
		return;
	}

	if (code.find("shared int") == std::string::npos) {
		std::cerr << "FAIL: shared int declaration not found" << std::endl;
		return;
	}

	std::cout << "PASS: Atomic operations with SharedMemory work" << std::endl;
}

// Test 4: WorkgroupReduce
void test_workgroup_reduce() {
	std::cout << "Test: WorkgroupReduce..." << std::endl;

	InspectorKernel1D inspector([](Int i) {
		SharedMemory<float, 256> shared;

		// Each thread contributes its global ID
		Expr<float>				 input	= ToFloat(i);

		// Perform reduction
		Expr<float>				 result = WorkgroupReduce(shared, input, Parallel::AddOp());

		// Only thread 0 stores the result
		Buffer<float>			 output(1);
		auto					 out	 = output.Bind();

		Int						 localId = LocalThreadId();
		If(localId == 0, [&]() { out[0] = result; });
	});

	std::string		  code = inspector.GetCode();

	// Check for shared memory and barrier calls
	if (code.find("shared float") == std::string::npos) {
		std::cerr << "FAIL: Shared memory not found" << std::endl;
		return;
	}

	// Should have multiple barrier calls for reduction
	int	   barrierCount = 0;
	size_t pos			= 0;
	while ((pos = code.find("barrier()", pos)) != std::string::npos) {
		++barrierCount;
		++pos;
	}

	if (barrierCount < 2) {
		std::cerr << "FAIL: Expected multiple barriers for reduction, found " << barrierCount << std::endl;
		return;
	}

	std::cout << "PASS: WorkgroupReduce generates correct code" << std::endl;
}

// Test 5: WorkgroupScanInclusive
void test_workgroup_scan() {
	std::cout << "Test: WorkgroupScanInclusive..." << std::endl;

	InspectorKernel1D inspector([](Int i) {
		SharedMemory<float, 256> shared;

		// Each thread contributes its global ID
		Expr<float>				 input	= ToFloat(i);

		// Perform inclusive scan
		Var<float>				 result = WorkgroupScanInclusive(shared, input, Parallel::AddOp());

		// Store result
		Buffer<float>			 output(256);
		auto					 out	 = output.Bind();

		Int						 localId = LocalThreadId();
		out[localId]					 = result;
	});

	std::string		  code = inspector.GetCode();

	// Check for shared memory
	if (code.find("shared float") == std::string::npos) {
		std::cerr << "FAIL: Shared memory not found" << std::endl;
		return;
	}

	std::cout << "PASS: WorkgroupScanInclusive generates correct code" << std::endl;
}

// Test 6: Full execution test - Counter using atomics
void test_atomic_counter_execution() {
	std::cout << "Test: Atomic Counter Execution..." << std::endl;

	// Simple test: each thread increments a counter
	constexpr int	 NUM_THREADS = 256;

	// Initialize counter with 0
	std::vector<int> zero(1, 0);
	Buffer<int>		 counter(zero);

	Kernel1D		 atomicKernel(
		[&](Int i) {
			auto cnt = counter.Bind();

			// Atomic increment
			ExprBase::NotUse(AtomicAdd(cnt[0], MakeInt(1)));
		},
		256); // Workgroup size 256

	// Dispatch single workgroup with 256 threads
	atomicKernel.Dispatch(1, true);

	// Download and verify
	std::vector<int> result(1);
	counter.Download(result);

	// Verify result
	if (result[0] == NUM_THREADS) {
		std::cout << "PASS: Atomic counter works correctly (count = " << result[0] << ")" << std::endl;
	} else {
		std::cerr << "FAIL: Expected count " << NUM_THREADS << ", got " << result[0] << std::endl;
	}
}

// Test 7: Parallel reduction execution
void test_reduction_execution() {
	std::cout << "Test: Parallel Reduction Execution..." << std::endl;

	// Create input data
	constexpr int	   NUM_ELEMENTS = 256;

	std::vector<float> inputData(NUM_ELEMENTS);
	for (int i = 0; i < NUM_ELEMENTS; ++i) {
		inputData[i] = 1.0f; // All ones, sum should be 256
	}

	Buffer<float> input(inputData);
	Buffer<float> output(1);

	Kernel1D	  reduceKernel(
		[&](Int i) {
			SharedMemory<float, 256> shared;

			// Load input
			auto					 in		 = input.Bind();
			Expr<float>				 val	 = in[i];

			// Perform reduction
			Expr<float>				 result	 = WorkgroupReduce(shared, val);

			// Store result (only thread 0)
			auto					 out	 = output.Bind();
			Int						 localId = LocalThreadId();
			If(localId == 0, [&]() { out[0] = result; });
		},
		256); // Workgroup size of 256

	// Dispatch single workgroup
	reduceKernel.Dispatch(1, true);

	// Download and verify
	std::vector<float> result(1);
	output.Download(result);

	if (std::abs(result[0] - NUM_ELEMENTS) < 0.01f) {
		std::cout << "PASS: Parallel reduction works correctly (sum = " << result[0] << ")" << std::endl;
	} else {
		std::cerr << "FAIL: Expected sum " << NUM_ELEMENTS << ", got " << result[0] << std::endl;
	}
}

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Testing SharedMemory, Atomics, and Parallel Primitives" << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_shared_memory_basic();
		test_atomic_operations();
		test_atomic_shared_memory();
		test_workgroup_reduce();
		test_workgroup_scan();
		test_atomic_counter_execution();
		test_reduction_execution();

		std::cout << "========================================" << std::endl;
		std::cout << "All tests completed!" << std::endl;
		std::cout << "========================================" << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "Test failed with exception: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
