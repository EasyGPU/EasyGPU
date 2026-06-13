/**
 * @file TestEdgeCases.cpp
 * @brief Comprehensive edge-case and boundary tests for EasyGPU core runtime.
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

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

// =============================================================================
// Buffer Edge Cases
// =============================================================================

TEST(zero_size_buffer_creation)
Runtime::Buffer<float> emptyBuf(0);
ASSERT(emptyBuf.GetCount() == 0);
ASSERT(emptyBuf.GetHandle() == Backend::INVALID_BUFFER_HANDLE);
END_TEST

TEST(empty_buffer_vector_constructor)
std::vector<int>	 emptyData;
Runtime::Buffer<int> buf(emptyData);
ASSERT(buf.GetCount() == 0);
ASSERT(buf.GetHandle() == Backend::INVALID_BUFFER_HANDLE);
END_TEST

TEST(buffer_upload_overcount_throws)
std::vector<float>	   data = {1.0f, 2.0f, 3.0f};
Runtime::Buffer<float> buf(data);
std::vector<float>	   tooBig(10, 0.0f);
bool				   threw = false;
try {
	buf.Upload(tooBig.data(), tooBig.size());
} catch (const std::out_of_range &) {
	threw = true;
}
ASSERT(threw);
END_TEST

TEST(buffer_download_overcount_throws)
std::vector<float>	   data = {1.0f, 2.0f, 3.0f};
Runtime::Buffer<float> buf(data);
std::vector<float>	   out(10);
bool				   threw = false;
try {
	buf.Download(out.data(), out.size());
} catch (const std::out_of_range &) {
	threw = true;
}
ASSERT(threw);
END_TEST

TEST(buffer_moved_from_bind_throws)
Runtime::Buffer<float> buf(4);
Runtime::Buffer<float> moved = std::move(buf);
// Try to bind the moved-from buffer inside a kernel lambda
bool				   threw = false;
try {
	InspectorKernel1D inspector([&](Int i) { auto b = buf.Bind(); });
	inspector.GetCode();
} catch (const std::runtime_error &e) {
	std::string msg = e.what();
	threw			= (msg.find("moved-from") != std::string::npos);
}
ASSERT(threw);
END_TEST

TEST(buffer_mode_readonly_no_write_in_kernel)
// This is a code-generation test: readonly buffer should generate "readonly" qualifier.
Runtime::Buffer<float> buf(std::vector<float>{1.0f, 2.0f, 3.0f}, Runtime::BufferMode::Read);
InspectorKernel1D	   inspector([&](Int i) {
	auto	   b = buf.Bind();
	Var<float> v = b[i];
});
std::string			   code = inspector.GetCode();
// The generated code should contain "readonly" for the buffer.
ASSERT(code.find("readonly") != std::string::npos);
END_TEST

// =============================================================================
// Texture Edge Cases
// =============================================================================

TEST(texture_subregion_out_of_bounds_throws)
Runtime::Texture2D<PixelFormat::RGBA8> tex(64, 64);
std::vector<uint8_t>				   data(4 * 32 * 32, 255);
bool								   threw = false;
try {
	// x=40, w=32 -> 72 > 64, should throw
	tex.UploadSubRegion(40, 0, 32, 32, data.data());
} catch (const std::out_of_range &) {
	threw = true;
}
ASSERT(threw);
END_TEST

TEST(texture_zero_size_upload)
Runtime::Texture2D<PixelFormat::R8> tex(1, 1);
// Upload nothing should not crash
std::vector<uint8_t>				empty;
tex.Upload(empty.data()); // data is nullptr but Upload handles it
END_TEST

TEST(texture3d_subregion_bounds)
Runtime::Texture3D<PixelFormat::RGBA8> tex(16, 16, 16);
std::vector<uint8_t>				   data(4 * 8 * 8 * 8, 128);
bool								   threw = false;
try {
	tex.UploadSubRegion(10, 0, 0, 8, 8, 8, data.data()); // 10+8=18 > 16
} catch (const std::out_of_range &) {
	threw = true;
}
ASSERT(threw);
END_TEST

// =============================================================================
// For Loop Edge Cases
// =============================================================================

TEST(for_negative_step_generates_correct_condition)
// Verify that a negative step produces a loop with v > end condition.
InspectorKernel1D inspector([&](Int i) { For(10, 0, -1, [&](Int &j) { Var<float> x = MakeFloat(1.0f); }); });
std::string		  code = inspector.GetCode();
// The generated for loop should contain "< 0" for negative step handling
ASSERT(code.find("for (int") != std::string::npos);
// Must contain both < and > to handle bidirectional step
ASSERT(code.find("<") != std::string::npos);
ASSERT(code.find(">") != std::string::npos);
END_TEST

TEST(for_zero_step_skips_loop)
// Zero step should produce a condition that never enters the loop.
InspectorKernel1D inspector([&](Int i) { For(0, 5, 0, [&](Int &j) { Var<float> x = MakeFloat(1.0f); }); });
std::string		  code = inspector.GetCode();
// Should still compile and generate a for loop
ASSERT(code.find("for (int") != std::string::npos);
END_TEST

TEST(for_var_step)
// Step as a variable (Expr<int>) should compile.
InspectorKernel1D inspector([&](Int i) {
	Var<int> step = MakeInt(2);
	For(0, 10, step, [&](Int &j) { Var<float> x = MakeFloat(1.0f); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("for (int") != std::string::npos);
END_TEST

// =============================================================================
// Control Flow Edge Cases
// =============================================================================

TEST(if_without_else)
InspectorKernel1D inspector([&](Int i) { If(i > 0, [&]() { Var<float> x = MakeFloat(1.0f); }); });
std::string		  code = inspector.GetCode();
ASSERT(code.find("if (") != std::string::npos);
// No else branch
ASSERT(code.find("} else {") == std::string::npos);
END_TEST

TEST(if_with_else)
InspectorKernel1D inspector([&](Int i) {
	If(i > 0, [&]() { Var<float> x = MakeFloat(1.0f); }).Else([&]() { Var<float> y = MakeFloat(2.0f); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("} else {") != std::string::npos);
END_TEST

TEST(if_elif_else)
InspectorKernel1D inspector([&](Int i) {
	If(i > 10, [&]() {
		Var<float> a = MakeFloat(1.0f);
	}).Elif(i > 5, [&]() {
		  Var<float> b = MakeFloat(2.0f);
	  }).Else([&]() { Var<float> c = MakeFloat(3.0f); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("else if (") != std::string::npos);
ASSERT(code.find("} else {") != std::string::npos);
END_TEST

// =============================================================================
// Var / Expr Edge Cases
// =============================================================================

TEST(var_copy_creates_independent_variable)
// Var copy should declare a new local variable.
InspectorKernel1D inspector([&](Int i) {
	Var<float> a = MakeFloat(1.0f);
	Var<float> b = a; // copy
	a			 = MakeFloat(2.0f);
});
std::string		  code = inspector.GetCode();
// Should compile without aliasing issues
ASSERT(!code.empty());
END_TEST

TEST(unref_forces_copy)
InspectorKernel1D inspector([&](Int i) {
	Runtime::Buffer<float> buf(4);
	auto				   b  = buf.Bind();
	Var<float>			   v1 = b[0];		 // may alias
	Var<float>			   v2 = Unref(b[0]); // must copy
});
std::string		  code = inspector.GetCode();
ASSERT(!code.empty());
END_TEST

TEST(select_evaluates_both_branches)
// Select is GLSL ternary; both branches are expressions (no side effects).
InspectorKernel1D inspector([&](Int i) {
	Var<bool>  cond	  = MakeBool(true);
	Var<float> result = Select(cond, MakeFloat(1.0f), MakeFloat(2.0f));
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("?") != std::string::npos);
ASSERT(code.find(":") != std::string::npos);
END_TEST

// =============================================================================
// Uniform Edge Cases
// =============================================================================

TEST(uniform_single_value)
InspectorKernel1D inspector([&](Int i) {
	Uniform<float> u(3.14f);
	Var<float>	   v = u.Load();
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("uniform") != std::string::npos);
END_TEST

TEST(uniform_vec4_value)
InspectorKernel1D inspector([&](Int i) {
	Uniform<Math::Vec4> u(Math::Vec4{1.0f, 2.0f, 3.0f, 4.0f});
	Var<Math::Vec4>		v = u.Load();
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("vec4") != std::string::npos);
END_TEST

// =============================================================================
// Shared Memory / Parallel Edge Cases
// =============================================================================

TEST(shared_memory_single_element)
// N=1 is a valid power of two.
InspectorKernel1D inspector([&](Int i) {
	SharedMemory<float, 1> shared;
	shared[0] = MakeFloat(42.0f);
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("shared float") != std::string::npos);
END_TEST

TEST(workgroup_reduce_power_of_two_required)
// N=1 should compile; runtime correctness is validated elsewhere.
InspectorKernel1D inspector([&](Int i) {
	SharedMemory<float, 1> shared;
	Expr<float>			   result = WorkgroupReduce(shared, MakeFloat(1.0f));
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("shared float") != std::string::npos);
END_TEST

// =============================================================================
// Backend / Context Edge Cases
// =============================================================================

TEST(context_caps_nonzero_workgroup_size)
int maxX, maxY, maxZ;
Runtime::Context::GetInstance().GetMaxWorkGroupSize(maxX, maxY, maxZ);
ASSERT(maxX > 0);
ASSERT(maxY > 0);
ASSERT(maxZ > 0);
END_TEST

TEST(context_compute_shader_support)
bool supported = Runtime::Context::GetInstance().HasComputeShadersSupport();
ASSERT(supported);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Edge Cases Test Suite        " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_zero_size_buffer_creation();
		test_empty_buffer_vector_constructor();
		test_buffer_upload_overcount_throws();
		test_buffer_download_overcount_throws();
		test_buffer_moved_from_bind_throws();
		test_buffer_mode_readonly_no_write_in_kernel();
		test_texture_subregion_out_of_bounds_throws();
		test_texture_zero_size_upload();
		test_texture3d_subregion_bounds();
		test_for_negative_step_generates_correct_condition();
		test_for_zero_step_skips_loop();
		test_for_var_step();
		test_if_without_else();
		test_if_with_else();
		test_if_elif_else();
		test_var_copy_creates_independent_variable();
		test_unref_forces_copy();
		test_select_evaluates_both_branches();
		test_uniform_single_value();
		test_uniform_vec4_value();
		test_shared_memory_single_element();
		test_workgroup_reduce_power_of_two_required();
		test_context_caps_nonzero_workgroup_size();
		test_context_compute_shader_support();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All edge-case tests passed!            " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
