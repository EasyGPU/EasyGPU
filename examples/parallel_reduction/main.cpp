/**
 * Parallel Reduction Example
 *
 * Demonstrates workgroup-level parallel reduction using shared memory.
 * This computes the sum of all elements in an array efficiently.
 */

#include <GPU.h>
#include <iostream>
#include <numeric>
#include <vector>

using namespace GPU;

int main() {
	// Configuration
	constexpr int	   NUM_ELEMENTS = 256; // Must match workgroup size for this simple example

	// Create input data (all 1.0f, so sum should be NUM_ELEMENTS)
	std::vector<float> inputData(NUM_ELEMENTS, 1.0f);

	// Create GPU buffers
	Buffer<float>	   input(inputData);
	Buffer<float>	   output(1); // Single element for result

	// Create reduction kernel
	// Workgroup size is 256, so we dispatch 1 workgroup
	Kernel1D		   reduceKernel(
		  [&](Int i) {
			  // Declare shared memory for workgroup
			  SharedMemory<float, 256> shared;

			  // Get local thread ID within workgroup (0-255)
			  Var<int>				   localId = LocalThreadId();

			  // Each thread loads one element
			  auto					   in	   = input.Bind();
			  auto					   out	   = output.Bind();

			  // Load input into shared memory
			  shared[localId]				   = in[i];

			  // Perform parallel reduction
			  // This computes the sum of all values in the workgroup
			  Expr<float> workgroupSum		   = WorkgroupReduce(shared, Expr<float>(shared[localId]));

			  // Only thread 0 writes the result
			  If(localId == 0, [&]() { out[0] = workgroupSum; });
		  },
		  256); // Workgroup size: 256 threads

	// Dispatch single workgroup
	std::cout << "Computing sum of " << NUM_ELEMENTS << " elements..." << std::endl;
	reduceKernel.Dispatch(1, true);

	// Read back result
	std::vector<float> result(1);
	output.Download(result);

	// Verify
	float expectedSum = static_cast<float>(NUM_ELEMENTS);
	std::cout << "GPU result: " << result[0] << std::endl;
	std::cout << "Expected:   " << expectedSum << std::endl;

	if (std::abs(result[0] - expectedSum) < 0.01f) {
		std::cout << "✓ Success! Parallel reduction works correctly." << std::endl;
		return 0;
	} else {
		std::cerr << "✗ Error: Results don't match!" << std::endl;
		return 1;
	}
}
