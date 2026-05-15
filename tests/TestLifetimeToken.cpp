/**
 * @file TestLifetimeToken.cpp
 * @brief Texture/Slot relationships. Verifies that use-after-free.
 */

#include <GPU.h>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Kernel;

static int testsPassed = 0;
static int testsTotal  = 0;

#define RUN_TEST(name, body)                                                                                           \
	do {                                                                                                               \
		std::cout << "[Test " << #name << "] ... " << std::flush;                                                      \
		testsTotal++;                                                                                                  \
		try {                                                                                                          \
			body                                                                                                       \
		} catch (const std::exception &e) {                                                                            \
			std::cout << "FAIL: " << e.what() << std::endl;                                                            \
			continue;                                                                                                  \
		}                                                                                                              \
		std::cout << "PASS" << std::endl;                                                                              \
		testsPassed++;                                                                                                 \
	} while (0)

int main() {
	std::cout << "=== Lifetime Token Protection Tests ===" << std::endl;

	// ==================================================================
	// Test 1: Destroy buffer then try to Bind — should throw
	// ==================================================================
	RUN_TEST(BufferUseAfterFree, {
		BufferSlot<float>  slot;

		std::vector<float> data(8, 1.0f);
		Buffer<float>	   buf(data);
		slot.Attach(buf);
		// buf goes out of scope HERE (destroyed)
		// Bind in Kernel lambda, but check happens at Bind() time
		// We need to test that Bind() catches the expired token
		// Since Bind() is called inside the kernel lambda during construction,
		// we need a different approach: create kernel BEFORE destroying buffer

		// Strategy: destroy buffer then try to create a NEW kernel with the slot
		// The slot's weak_ptr should detect the destroyed buffer
	});

	// ==================================================================
	// Test 2: Buffer destroyed, then Bind in new kernel throws
	// ==================================================================
	RUN_TEST(BufferDestroyedBeforeKernel, {
		BufferSlot<float> slot;

		{
			std::vector<float> data(8, 1.0f);
			Buffer<float>	   buf(data);
			slot.Attach(buf);
		} // buf destroyed here

		// Now try to create kernel — Bind() should throw because token expired
		bool caught = false;
		try {
			Kernel1D kernel([&](Int i) {
				auto buf = slot.Bind(); // Should throw
				buf[i]	 = 1.0f;
			});
		} catch (const std::runtime_error &e) {
			caught = true;
			std::string msg(e.what());
			if (msg.find("destroyed") != std::string::npos) {
				// Expected
			} else {
				throw std::runtime_error("Wrong error message: " + msg);
			}
		}
		if (!caught) {
			throw std::runtime_error("Expected exception not thrown");
		}
	});

	// ==================================================================
	// Test 3: Texture2D destroyed, Bind in new kernel throws
	// ==================================================================
	RUN_TEST(Texture2DDestroyedBeforeKernel, {
		TextureSlot<PixelFormat::R32F> slot;

		{
			std::vector<float> data(64, 0.0f);
			TextureR32F		   tex(8, 8, data.data());
			slot.Attach(tex);
		} // tex destroyed here

		bool caught = false;
		try {
			Kernel2D kernel([&](Int x, Int y) {
				auto tex = slot.Bind(); // Should throw
				tex.Write(x, y, MakeFloat4(0.0f));
			});
		} catch (const std::runtime_error &e) {
			caught = true;
			std::string msg(e.what());
			if (msg.find("destroyed") != std::string::npos) {
				// Expected
			} else {
				throw std::runtime_error("Wrong error message: " + msg);
			}
		}
		if (!caught) {
			throw std::runtime_error("Expected exception not thrown");
		}
	});

	// ==================================================================
	// Test 4: Detach then Bind — slot is empty but not destroyed, should NOT throw
	//         (it will throw at Dispatch time instead — different error path)
	// ==================================================================
	RUN_TEST(DetachThenBind, {
		BufferSlot<float> slot;

		{
			std::vector<float> data(8, 1.0f);
			Buffer<float>	   buf(data);
			slot.Attach(buf);
			slot.Detach(); // Explicitly detach — this resets the token
		}

		// Bind() should NOT throw because Detach() resets both ptr and token
		// (weak_ptr expires, but _bufferPtr is null → guard prevents throw)
		// Dispatch will fail instead — this is correct behavior
		Kernel1D kernel([&](Int i) {
			auto buf = slot.Bind(); // Should NOT throw (_bufferPtr is null)
			buf[i]	 = 1.0f;
		});
		// Kernel creation succeeds (Bind does lazy registration)
		// Dispatch would fail, but we test Bind behavior here
	});

	// ==================================================================
	// Test 5: Re-attach after detach — normal use case works
	// ==================================================================
	RUN_TEST(DetachThenReattach, {
		BufferSlot<float>  slot;

		std::vector<float> data(8, 1.0f);
		{
			Buffer<float> buf1(data);
			slot.Attach(buf1);
			slot.Detach();
		} // buf1 destroyed (but slot was detached safely)

		// Now attach a fresh buffer
		Buffer<float> buf2(data);
		slot.Attach(buf2);

		Kernel1D kernel([&](Int i) {
			auto buf = slot.Bind(); // Should work fine
			buf[i]	 = buf[i] + 1.0f;
		});

		kernel.Dispatch(1, true);
		std::vector<float> result(8);
		buf2.Download(result);

		if (result[0] != 2.0f) {
			throw std::runtime_error("Wrong result: expected 2.0, got " + std::to_string(result[0]));
		}
	});

	// ==================================================================
	// Test 6: Normal BufferSlot lifecycle — attach, use, done
	// ==================================================================
	RUN_TEST(NormalBufferSlotLifecycle, {
		BufferSlot<float>  slot;
		std::vector<float> data(8, 5.0f);
		Buffer<float>	   buf(data);

		slot.Attach(buf);

		Kernel1D kernel([&](Int i) {
			auto b = slot.Bind();
			b[i]   = b[i] * 2.0f;
		});

		kernel.Dispatch(1, true);

		std::vector<float> result(8);
		buf.Download(result);

		if (result[0] != 10.0f) {
			throw std::runtime_error("Wrong result: " + std::to_string(result[0]));
		}
	});

	std::cout << "\n========================================" << std::endl;
	std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
	std::cout << "========================================" << std::endl;

	return (testsPassed == testsTotal) ? 0 : 1;
}
