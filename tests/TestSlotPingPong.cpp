/**
 * TestSlotPingPong.cpp:
 *      @Descripiton    :   Ping-pong algorithm tests using BufferSlot
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <GPU.h>
#include <iostream>
#include <vector>

bool FloatEq(float a, float b, float epsilon = 0.01f) {
	return std::abs(a - b) < epsilon;
}

int main() {
	try {
		std::cout << "=== Slot Ping-Pong Tests ===" << std::endl;
		int testsPassed = 0;
		int testsTotal	= 0;

		// ==================================================================
		// Test 1: Simple ping-pong (Jacobi iteration)
		// ==================================================================
		{
			std::cout << "\n[Test 1] Jacobi iteration ping-pong..." << std::flush;
			testsTotal++;

			BufferSlot<float>  readSlot;
			BufferSlot<float>  writeSlot;

			// Jacobi iteration: new[i] = (old[i-1] + old[i] + old[i+1]) / 3
			Kernel1D		   jacobi([&](Int i) {
				  auto	src	  = readSlot.Bind();
				  auto	dst	  = writeSlot.Bind();

				  // Handle boundaries with Clamp
				  Int	left  = Max(i - 1, 0);
				  Int	right = Min(i + 1, 63);

				  Float sum	  = src[left] + src[i] + src[right];
				  dst[i]	  = sum / 3.0f;
			  });

			// Initial data: step function
			std::vector<float> ping(64);
			std::vector<float> pong(64);
			for (int i = 0; i < 64; ++i) {
				ping[i] = (i < 32) ? 0.0f : 100.0f;
			}

			Buffer<float> pingBuf(ping);
			Buffer<float> pongBuf(pong);

			// Run 10 iterations
			for (int iter = 0; iter < 10; ++iter) {
				if (iter % 2 == 0) {
					readSlot.Attach(pingBuf);
					writeSlot.Attach(pongBuf);
				} else {
					readSlot.Attach(pongBuf);
					writeSlot.Attach(pingBuf);
				}
				jacobi.Dispatch(1, true);
			}

			// Read final result
			std::vector<float> result(64);
			if (10 % 2 == 0) {
				pingBuf.Download(result);
			} else {
				pongBuf.Download(result);
			}

			// Check that values have diffused
			// After smoothing, the step should be less sharp
			bool pass = (result[31] > 20.0f && result[31] < 80.0f) && (result[32] > 20.0f && result[32] < 80.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (values not diffused: " << result[31] << ", " << result[32] << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 2: Multi-pass blur
		// ==================================================================
		{
			std::cout << "[Test 2] Multi-pass box blur..." << std::flush;
			testsTotal++;

			BufferSlot<Vec4>  readSlot;
			BufferSlot<Vec4>  writeSlot;

			Kernel2D		  blur([&](Int x, Int y) {
				 auto	src	   = readSlot.Bind();
				 auto	dst	   = writeSlot.Bind();

				 Int	width  = MakeInt(32);
				 Int	height = MakeInt(32);

				 // 3x3 box blur
				 Float4 sum	   = MakeFloat4(0.0f);
				 Int	count  = MakeInt(0);

				 For(-1, 2, [&](Int &dy) {
					 For(-1, 2, [&](Int &dx) {
						 Int sx	 = Clamp(x + dx, 0, width - 1);
						 Int sy	 = Clamp(y + dy, 0, height - 1);
						 Int idx = sy * width + sx;
						 sum	 = sum + src[idx];
						 count	 = count + 1;
					 });
				 });

				 Int idx  = y * width + x;
				 dst[idx] = sum / ToFloat(count);
			 });

			// Create test image with a bright spot in center
			std::vector<Vec4> ping(32 * 32, Vec4(0, 0, 0, 1));
			std::vector<Vec4> pong(32 * 32, Vec4(0, 0, 0, 1));
			for (int y = 14; y < 18; ++y) {
				for (int x = 14; x < 18; ++x) {
					ping[y * 32 + x] = Vec4(100, 100, 100, 1);
				}
			}

			Buffer<Vec4> pingBuf(ping);
			Buffer<Vec4> pongBuf(pong);

			// Run 5 blur passes
			for (int iter = 0; iter < 5; ++iter) {
				if (iter % 2 == 0) {
					readSlot.Attach(pingBuf);
					writeSlot.Attach(pongBuf);
				} else {
					readSlot.Attach(pongBuf);
					writeSlot.Attach(pingBuf);
				}
				blur.Dispatch(2, 2, true);
			}

			// Read result
			std::vector<Vec4> result(32 * 32);
			if (5 % 2 == 0) {
				pingBuf.Download(result);
			} else {
				pongBuf.Download(result);
			}

			// After blurring, center should still be brightest
			float centerVal = result[16 * 32 + 16].x;
			float edgeVal	= result[0].x;
			bool  pass		= (centerVal > edgeVal);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (center=" << centerVal << " not > edge=" << edgeVal << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 3: Gauss-Seidel with texture slots
		// ==================================================================
		{
			std::cout << "[Test 3] Texture slot ping-pong..." << std::flush;
			testsTotal++;

			TextureSlot<PixelFormat::R32F> readSlot;
			TextureSlot<PixelFormat::R32F> writeSlot;

			Kernel2D					   relax([&](Int x, Int y) {
				  auto	src = readSlot.Bind();
				  auto	dst = writeSlot.Bind();

				  // Average of neighbors
				  Float sum = MakeFloat(0.0f);
				  sum		= sum + src.Read(Clamp(x - 1, 0, 15), y).x();
				  sum		= sum + src.Read(Clamp(x + 1, 0, 15), y).x();
				  sum		= sum + src.Read(x, Clamp(y - 1, 0, 15)).x();
				  sum		= sum + src.Read(x, Clamp(y + 1, 0, 15)).x();

				  dst.Write(x, y, MakeFloat4(sum / 4.0f, 0.0f, 0.0f, 1.0f));
			  });

			// Create checkerboard pattern
			std::vector<float>			   ping(16 * 16, 0.0f);
			std::vector<float>			   pong(16 * 16, 0.0f);
			for (int y = 0; y < 16; ++y) {
				for (int x = 0; x < 16; ++x) {
					ping[y * 16 + x] = ((x + y) % 2 == 0) ? 100.0f : 0.0f;
				}
			}

			Texture2D<PixelFormat::R32F> pingTex(16, 16, ping.data());
			Texture2D<PixelFormat::R32F> pongTex(16, 16, pong.data());

			// Run 8 iterations
			for (int iter = 0; iter < 8; ++iter) {
				if (iter % 2 == 0) {
					readSlot.Attach(pingTex);
					writeSlot.Attach(pongTex);
				} else {
					readSlot.Attach(pongTex);
					writeSlot.Attach(pingTex);
				}
				relax.Dispatch(1, 1, true);
			}

			// Check result - values should be smoother
			std::vector<float> result(16 * 16 * 4);
			if (8 % 2 == 0) {
				pingTex.Download(result);
			} else {
				pongTex.Download(result);
			}

			// Difference between adjacent pixels should be smaller
			float diff1 = std::abs(result[0] - result[4]);	// (0,0) vs (1,0)
			float diff2 = std::abs(result[0] - result[64]); // (0,0) vs (0,1)
			bool  pass	= (diff1 < 50.0f) && (diff2 < 50.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (still too sharp: diff1=" << diff1 << ", diff2=" << diff2 << ")" << std::endl;
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
