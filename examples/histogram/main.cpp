/**
 * @file main.cpp
 * @brief Histogram Example.
 */

#include <GPU.h>
#include <iostream>
#include <vector>

using namespace GPU;

int main() {
	// Configuration
	constexpr int	   NUM_ELEMENTS = 10000;
	constexpr int	   NUM_BINS		= 10;

	// Create input data (values 0.0 to 1.0)
	std::vector<float> inputData(NUM_ELEMENTS);
	for (int i = 0; i < NUM_ELEMENTS; ++i) {
		inputData[i] = static_cast<float>(i % NUM_BINS) / NUM_BINS;
	}

	// Create GPU buffers
	Buffer<float>	 input(inputData);

	// Initialize histogram with zeros
	std::vector<int> zeroBins(NUM_BINS, 0);
	Buffer<int>		 histogram(zeroBins);

	// Create histogram kernel
	Kernel1D		 histogramKernel([&](Int i) {
		// Bounds check - only process valid elements
		If(i < NUM_ELEMENTS, [&]() {
			auto  in	= input.Bind();
			auto  hist	= histogram.Bind();

			// Read input value (0.0 to 1.0)
			Float value = in[i];

			// Compute bin index (0 to NUM_BINS-1)
			Int	  bin	= Clamp(ToInt(value * MakeFloat(NUM_BINS)), 0, NUM_BINS - 1);

			// Atomically increment the bin
			// ExprBase::NotUse() is used to discard the return value (old count)
			ExprBase::NotUse(AtomicAdd(hist[bin], MakeInt(1)));
		});
	});

	// Dispatch
	int				 numGroups = (NUM_ELEMENTS + 255) / 256;
	std::cout << "Computing histogram of " << NUM_ELEMENTS << " elements into " << NUM_BINS << " bins..." << std::endl;
	histogramKernel.Dispatch(numGroups, true);

	// Read back histogram
	std::vector<int> result(NUM_BINS);
	histogram.Download(result);

	// Verify results
	int	 expectedPerBin = NUM_ELEMENTS / NUM_BINS;
	bool correct		= true;

	std::cout << "\nHistogram results:" << std::endl;
	std::cout << "Bin\tCount\tExpected" << std::endl;
	std::cout << "---\t-----\t--------" << std::endl;

	for (int i = 0; i < NUM_BINS; ++i) {
		std::cout << i << "\t" << result[i] << "\t" << expectedPerBin;
		if (result[i] != expectedPerBin) {
			std::cout << " X";
			correct = false;
		} else {
			std::cout << " OK";
		}
		std::cout << std::endl;
	}

	if (correct) {
		std::cout << "\n✓ Success! Histogram computation works correctly." << std::endl;
		return 0;
	} else {
		std::cerr << "\n✗ Error: Some bins have incorrect counts!" << std::endl;
		return 1;
	}
}
