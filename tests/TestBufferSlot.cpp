/**
 * TestBufferSlot.cpp:
 *      @Descripiton    :   BufferSlot functionality tests
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <GPU.h>
#include <cmath>
#include <iostream>
#include <vector>

bool FloatEq(float a, float b, float epsilon = 0.001f) {
	return std::abs(a - b) < epsilon;
}

int main() {
	try {
		std::cout << "=== BufferSlot Tests ===" << std::endl;
		int testsPassed = 0;
		int testsTotal	= 0;

		// ==================================================================
		// Test 1: Basic BufferSlot Attach/Detach
		// ==================================================================
		{
			std::cout << "\n[Test 1] Attach/Detach functionality..." << std::flush;
			testsTotal++;

			BufferSlot<float>  slot;
			std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
			Buffer<float>	   buf(data);

			if (slot.IsAttached()) {
				std::cout << " FAIL (should not be attached initially)" << std::endl;
			} else {
				slot.Attach(buf);
				if (!slot.IsAttached()) {
					std::cout << " FAIL (should be attached after Attach)" << std::endl;
				} else if (slot.GetAttached() != &buf) {
					std::cout << " FAIL (GetAttached returned wrong pointer)" << std::endl;
				} else {
					slot.Detach();
					if (slot.IsAttached()) {
						std::cout << " FAIL (should not be attached after Detach)" << std::endl;
					} else {
						std::cout << " PASS" << std::endl;
						testsPassed++;
					}
				}
			}
		}

		// ==================================================================
		// Test 2: Basic kernel with slot
		// ==================================================================
		{
			std::cout << "[Test 2] Basic kernel execution..." << std::flush;
			testsTotal++;

			BufferSlot<float>  inputSlot;
			BufferSlot<float>  outputSlot;

			Kernel1D		   kernel([&](Int i) {
				  auto in  = inputSlot.Bind();
				  auto out = outputSlot.Bind();
				  out[i]   = in[i] * 2.0f;
			  });

			std::vector<float> input	= {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
			std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
			std::vector<float> result(5);

			Buffer<float>	   inputBuf(input);
			Buffer<float>	   outputBuf(5);

			inputSlot.Attach(inputBuf);
			outputSlot.Attach(outputBuf);
			kernel.Dispatch(1, true);

			outputBuf.Download(result);

			bool pass = true;
			for (size_t i = 0; i < 5; ++i) {
				if (!FloatEq(result[i], expected[i])) {
					pass = false;
					break;
				}
			}

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL" << std::endl;
			}
		}

		// ==================================================================
		// Test 3: Switch buffers without recompilation
		// ==================================================================
		{
			std::cout << "[Test 3] Buffer switching (no recompilation)..." << std::flush;
			testsTotal++;

			BufferSlot<float>  inputSlot;
			BufferSlot<float>  outputSlot;

			Kernel1D		   kernel([&](Int i) {
				  auto in  = inputSlot.Bind();
				  auto out = outputSlot.Bind();
				  out[i]   = in[i] + 10.0f;
			  });

			std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
			std::vector<float> data2 = {100.0f, 200.0f, 300.0f};
			std::vector<float> result(3);

			Buffer<float>	   buf1(data1);
			Buffer<float>	   buf2(data2);
			Buffer<float>	   outBuf(3);

			outputSlot.Attach(outBuf);

			// First dispatch
			inputSlot.Attach(buf1);
			kernel.Dispatch(1, true);
			outBuf.Download(result);
			bool pass1 = FloatEq(result[0], 11.0f) && FloatEq(result[1], 12.0f);

			// Second dispatch - switch buffer
			inputSlot.Attach(buf2);
			kernel.Dispatch(1, true);
			outBuf.Download(result);
			bool pass2 = FloatEq(result[0], 110.0f) && FloatEq(result[1], 210.0f);

			if (pass1 && pass2) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL" << std::endl;
			}
		}

		// ==================================================================
		// Test 4: Ping-pong with slots
		// ==================================================================
		{
			std::cout << "[Test 4] Ping-pong buffer swapping..." << std::flush;
			testsTotal++;

			BufferSlot<float>  readSlot;
			BufferSlot<float>  writeSlot;

			Kernel1D		   accumulate([&](Int i) {
				  auto src = readSlot.Bind();
				  auto dst = writeSlot.Bind();
				  dst[i]   = src[i] + 1.0f;
			  });

			std::vector<float> pingData(5, 0.0f);
			std::vector<float> pongData(5, 0.0f);

			Buffer<float>	   ping(pingData);
			Buffer<float>	   pong(pongData);

			// Iteration 1: ping -> pong
			readSlot.Attach(ping);
			writeSlot.Attach(pong);
			accumulate.Dispatch(1, true);

			// Iteration 2: pong -> ping
			readSlot.Attach(pong);
			writeSlot.Attach(ping);
			accumulate.Dispatch(1, true);

			std::vector<float> result(5);
			ping.Download(result);

			bool pass = FloatEq(result[0], 2.0f);
			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", expected 2.0)" << std::endl;
			}
		}

		// ==================================================================
		// Summary
		// ==================================================================
		std::cout << "\n========================================" << std::endl;
		std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
		std::cout << "========================================" << std::endl;

		return (testsPassed == testsTotal) ? 0 : 1;

	} catch (const std::exception &e) {
		std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
		return 1;
	}
}
