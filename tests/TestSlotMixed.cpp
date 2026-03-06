/**
 * TestSlotMixed.cpp:
 *      @Descripiton    :   Mixed usage tests for Slots with other features
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
		std::cout << "=== Slot Mixed Usage Tests ===" << std::endl;
		int testsPassed = 0;
		int testsTotal	= 0;

		// ==================================================================
		// Test 1: BufferSlot + TextureSlot together
		// ==================================================================
		{
			std::cout << "\n[Test 1] BufferSlot + TextureSlot..." << std::flush;
			testsTotal++;

			BufferSlot<float>			   bufferSlot;
			TextureSlot<PixelFormat::R32F> textureSlot;

			Kernel2D					   kernel([&](Int x, Int y) {
				  auto	buf	   = bufferSlot.Bind();
				  auto	tex	   = textureSlot.Bind();

				  // Read from buffer and texture
				  Int	idx	   = y * 16 + x;
				  Float bufVal = buf[idx];
				  Float texVal = tex.Read(x, y).x();

				  // Write sum back to texture
				  tex.Write(x, y, MakeFloat4(bufVal + texVal, 0.0f, 0.0f, 1.0f));
			  });

			// Setup buffer
			std::vector<float>			   bufData(16 * 16);
			for (int i = 0; i < 16 * 16; ++i)
				bufData[i] = static_cast<float>(i);
			Buffer<float>	   buffer(bufData);

			// Setup texture
			std::vector<float> texData(16 * 16, 10.0f);
			TextureR32F		   texture(16, 16, texData.data());

			bufferSlot.Attach(buffer);
			textureSlot.Attach(texture);
			kernel.Dispatch(1, 1, true);

			std::vector<float> result(16 * 16 * 4);
			texture.Download(result);

			// Pixel (0,0): buf[0]=0 + tex=10 = 10
			// Pixel (1,0): buf[1]=1 + tex=10 = 11
			// Note: For R32F format, each pixel is 1 float, so result[1] is pixel (1,0)
			bool pass = FloatEq(result[0], 10.0f) && FloatEq(result[1], 11.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", " << result[4] << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 2: Slot + Uniform together
		// ==================================================================
		{
			std::cout << "[Test 2] BufferSlot + Uniform..." << std::flush;
			testsTotal++;

			BufferSlot<float>  slot;
			Uniform<float>	   scale;
			Uniform<float>	   offset;

			Kernel1D		   kernel([&](Int i) {
				  auto	buf = slot.Bind();
				  Float s	= scale.Load();
				  Float o	= offset.Load();
				  buf[i]	= buf[i] * s + o;
			  });

			std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
			Buffer<float>	   buffer(data);
			slot.Attach(buffer);

			// First dispatch: scale=2, offset=10
			scale  = 2.0f;
			offset = 10.0f;
			kernel.Dispatch(1, true);

			std::vector<float> result(5);
			buffer.Download(result);
			bool pass1 = FloatEq(result[0], 12.0f) && FloatEq(result[4], 20.0f);

			// Second dispatch: scale=0.5, offset=0 (no recompilation!)
			scale	   = 0.5f;
			offset	   = 0.0f;
			kernel.Dispatch(1, true);

			buffer.Download(result);
			// Previous results: 12, 14, 16, 18, 20 -> *0.5 = 6, 7, 8, 9, 10
			bool pass2 = FloatEq(result[0], 6.0f) && FloatEq(result[4], 10.0f);

			if (pass1 && pass2) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (pass1=" << pass1 << ", pass2=" << pass2 << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 3: Slot + static Buffer together
		// ==================================================================
		{
			std::cout << "[Test 3] BufferSlot + static Buffer..." << std::flush;
			testsTotal++;

			// Static buffer (traditional binding)
			std::vector<float> constData = {100.0f, 200.0f, 300.0f};
			Buffer<float>	   constBuffer(constData);

			// Slot buffer (dynamic binding)
			BufferSlot<float>  slot;

			Kernel1D		   kernel([&](Int i) {
				  // Access static buffer
				  auto c = constBuffer.Bind();
				  // Access slot buffer
				  auto s = slot.Bind();

				  s[i]	 = s[i] + c[i];
			  });

			std::vector<float> data = {1.0f, 2.0f, 3.0f};
			Buffer<float>	   buffer(data);
			slot.Attach(buffer);

			kernel.Dispatch(1, true);

			std::vector<float> result(3);
			buffer.Download(result);

			// 1+100=101, 2+200=202, 3+300=303
			bool pass = FloatEq(result[0], 101.0f) && FloatEq(result[1], 202.0f) && FloatEq(result[2], 303.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", " << result[1] << ", " << result[2] << ")" << std::endl;
			}
		}

		// ==================================================================
		// Test 4: Multiple dispatches with slot + uniform changes
		// ==================================================================
		{
			std::cout << "[Test 4] Multiple dispatches with slot+uniform changes..." << std::flush;
			testsTotal++;

			BufferSlot<float>  slot;
			Uniform<int>	   iteration;

			Kernel1D		   kernel([&](Int i) {
				  auto buf	= slot.Bind();
				  Int  iter = iteration.Load();
				  buf[i]	= buf[i] + ToFloat(iter);
			  });

			std::vector<float> data(10, 0.0f);
			Buffer<float>	   buffer(data);
			slot.Attach(buffer);

			// Run 5 iterations
			for (int iter = 1; iter <= 5; ++iter) {
				iteration = iter;
				kernel.Dispatch(1, true);
			}

			std::vector<float> result(10);
			buffer.Download(result);

			// Sum should be 1+2+3+4+5 = 15
			bool pass = FloatEq(result[0], 15.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", expected 15.0)" << std::endl;
			}
		}

		// ==================================================================
		// Test 5: BufferSlot + TextureSlot together
		// ==================================================================
		{
			std::cout << "[Test 5] BufferSlot + TextureSlot..." << std::flush;
			testsTotal++;

			BufferSlot<float>			   bufSlot;
			TextureSlot<PixelFormat::R32F> tex2DSlot;

			Kernel1D					   kernel([&](Int i) {
				  auto buf = bufSlot.Bind();
				  // Note: Can't use 2D texture Bind in 1D kernel easily,
				  // so we just verify the buffer slot works
				  buf[i]   = buf[i] * 2.0f;
			  });

			std::vector<float>			   data(8, 5.0f);
			Buffer<float>				   buffer(data);
			bufSlot.Attach(buffer);

			// Also attach texture slot (even though not used in this kernel,
			// it should not interfere)
			std::vector<float> tex2DData(16 * 16, 1.0f);
			TextureR32F		   tex2D(16, 16, tex2DData.data());
			tex2DSlot.Attach(tex2D);

			kernel.Dispatch(1, true);

			std::vector<float> result(8);
			buffer.Download(result);

			bool pass = FloatEq(result[0], 10.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", expected 10.0)" << std::endl;
			}
		}

		// ==================================================================
		// Test 6: Slot with Callable
		// ==================================================================
		{
			std::cout << "[Test 6] BufferSlot with Callable..." << std::flush;
			testsTotal++;

			BufferSlot<float>	   slot;

			Callable<Float(Float)> Process = [](Float &x) { Return(x * x + 1.0f); };

			Kernel1D			   kernel([&](Int i) {
				  auto buf = slot.Bind();
				  buf[i]   = Process(buf[i]);
			  });

			std::vector<float>	   data = {1.0f, 2.0f, 3.0f, 4.0f};
			Buffer<float>		   buffer(data);
			slot.Attach(buffer);

			kernel.Dispatch(1, true);

			std::vector<float> result(4);
			buffer.Download(result);

			// 1*1+1=2, 2*2+1=5, 3*3+1=10, 4*4+1=17
			bool pass = FloatEq(result[0], 2.0f) && FloatEq(result[1], 5.0f) && FloatEq(result[2], 10.0f) &&
						FloatEq(result[3], 17.0f);

			if (pass) {
				std::cout << " PASS" << std::endl;
				testsPassed++;
			} else {
				std::cout << " FAIL (got " << result[0] << ", " << result[1] << ")" << std::endl;
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
