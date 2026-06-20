/**
 * @file TestBufferSlot.cpp
 * @brief BufferSlot functionality tests.
 */

#include <GPU.h>
#include <cmath>
#include <iostream>
#include <vector>

bool FloatEq(float a, float b, float epsilon = 0.001f) {
	return std::abs(a - b) < epsilon;
}

EASYGPU_STRUCT(BufferSlotParticle, (GPU::Math::Vec4, position), (GPU::Math::Vec4, velocity), (float, mass));

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
				auto in	 = inputSlot.Bind();
				auto out = outputSlot.Bind();
				out[i]	 = in[i] * 2.0f;
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
				auto in	 = inputSlot.Bind();
				auto out = outputSlot.Bind();
				out[i]	 = in[i] + 10.0f;
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
				dst[i]	 = src[i] + 1.0f;
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
		// Test 5: Multiple slots in same kernel
		// ==================================================================
		{
			std::cout << "[Test 5] Multiple slots in same kernel..." << std::flush;
			testsTotal++;

			BufferSlot<float>  slotA;
			BufferSlot<float>  slotB;
			BufferSlot<float>  slotC;

			Kernel1D		   kernel([&](Int i) {
				auto a = slotA.Bind();
				auto b = slotB.Bind();
				auto c = slotC.Bind();
				c[i]   = a[i] + b[i];
			});

			std::vector<float> dataA = {1.0f, 2.0f, 3.0f};
			std::vector<float> dataB = {10.0f, 20.0f, 30.0f};
			std::vector<float> result(3);

			Buffer<float>	   bufA(dataA);
			Buffer<float>	   bufB(dataB);
			Buffer<float>	   bufC(3);

			slotA.Attach(bufA);
			slotB.Attach(bufB);
			slotC.Attach(bufC);

			kernel.Dispatch(1, true);
			bufC.Download(result);

			bool pass = FloatEq(result[0], 11.0f) && FloatEq(result[1], 22.0f) && FloatEq(result[2], 33.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", " << result[1] << ", " << result[2] << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 6: Large data processing
		// ==================================================================
		{
			std::cout << "[Test 6] Large data processing (1M elements)..." << std::flush;
			testsTotal++;

			const size_t	   N = 1024 * 1024;
			BufferSlot<float>  inputSlot;
			BufferSlot<float>  outputSlot;

			Kernel1D		   kernel([&](Int i) {
				auto in	 = inputSlot.Bind();
				auto out = outputSlot.Bind();
				out[i]	 = in[i] * 3.14159f;
			});

			std::vector<float> input(N);
			for (size_t i = 0; i < N; ++i)
				input[i] = static_cast<float>(i);
			std::vector<float> result(N);

			Buffer<float>	   inputBuf(input);
			Buffer<float>	   outputBuf(N);

			inputSlot.Attach(inputBuf);
			outputSlot.Attach(outputBuf);

			kernel.Dispatch((N + 255) / 256, true);
			outputBuf.Download(result);

			bool		 pass	   = true;
			// Spot check
			const size_t indices[] = {0, 1, 100, 10000, N / 2, N - 1};
			for (size_t i : indices) {
				if (!FloatEq(result[i], input[i] * 3.14159f, 0.01f)) {
					pass = false;
					std::cout << "\n  Mismatch at " << i << ": got " << result[i] << ", expected "
							  << input[i] * 3.14159f;
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
		// Test 7: Vector type BufferSlot
		// ==================================================================
		{
			std::cout << "[Test 7] Vec4 BufferSlot..." << std::flush;
			testsTotal++;

			BufferSlot<Vec4>  inputSlot;
			BufferSlot<Vec4>  outputSlot;

			Kernel1D		  kernel([&](Int i) {
				auto in	 = inputSlot.Bind();
				auto out = outputSlot.Bind();
				auto v	 = in[i];
				out[i]	 = MakeFloat4(v.x() * 2.0f, v.y() * 2.0f, v.z() * 2.0f, v.w() * 2.0f);
			});

			std::vector<Vec4> input = {Vec4(1.0f, 2.0f, 3.0f, 4.0f), Vec4(5.0f, 6.0f, 7.0f, 8.0f)};
			std::vector<Vec4> result(2);

			Buffer<Vec4>	  inputBuf(input);
			Buffer<Vec4>	  outputBuf(2);

			inputSlot.Attach(inputBuf);
			outputSlot.Attach(outputBuf);
			kernel.Dispatch(1, true);
			outputBuf.Download(result);

			bool pass = FloatEq(result[0].x, 2.0f) && FloatEq(result[0].y, 4.0f) && FloatEq(result[0].z, 6.0f) &&
						FloatEq(result[0].w, 8.0f) && FloatEq(result[1].x, 10.0f) && FloatEq(result[1].y, 12.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL" << std::endl;
			}
		}

		// ==================================================================
		// Test 8: EASYGPU_STRUCT BufferSlot
		// ==================================================================
		{
			std::cout << "[Test 8] EASYGPU_STRUCT BufferSlot..." << std::flush;
			testsTotal++;

			BufferSlot<BufferSlotParticle> inputSlot;
			BufferSlot<BufferSlotParticle> outputSlot;

			Kernel1D kernel([&](Int i) {
				auto in	 = inputSlot.Bind();
				auto out = outputSlot.Bind();
				auto p	 = in[i];
				p.mass() = p.mass() + 1.0f;
				out[i]	 = p;
			});

			std::vector<BufferSlotParticle> input(2);
			input[0].position = Vec4(1.0f, 2.0f, 3.0f, 1.0f);
			input[0].velocity = Vec4(0.5f, 0.0f, 0.0f, 0.0f);
			input[0].mass	  = 2.0f;
			input[1].position = Vec4(4.0f, 5.0f, 6.0f, 1.0f);
			input[1].velocity = Vec4(0.0f, 0.5f, 0.0f, 0.0f);
			input[1].mass	  = 3.0f;

			std::vector<BufferSlotParticle> result(2);
			Buffer<BufferSlotParticle>	  inputBuf(input);
			Buffer<BufferSlotParticle>	  outputBuf(2);

			inputSlot.Attach(inputBuf);
			outputSlot.Attach(outputBuf);
			kernel.Dispatch(1, true);
			outputBuf.Download(result);

			bool pass = FloatEq(result[0].mass, 3.0f) && FloatEq(result[1].mass, 4.0f) &&
						FloatEq(result[0].position.x, 1.0f) && FloatEq(result[1].velocity.y, 0.5f);
			if (std::string(inputSlot.GetTypeName()) != "BufferSlotParticle") {
				pass = false;
			}

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL" << std::endl;
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
