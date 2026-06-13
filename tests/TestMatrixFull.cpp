/**
 * @file TestMatrixFull.cpp
 * @brief Comprehensive tests for matrix types: construction, multiplication,.
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

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

// =============================================================================
// Code Generation: Matrix Construction
// =============================================================================

TEST(mat2_construction_codegen)
InspectorKernel1D inspector([&](Int i) {
	Var<Vec2> c0 = MakeFloat2(1.0f, 0.0f);
	Var<Vec2> c1 = MakeFloat2(0.0f, 1.0f);
	Var<Mat2> m	 = MakeMat2(c0, c1);
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("mat2") != std::string::npos);
END_TEST

TEST(mat3_construction_codegen)
InspectorKernel1D inspector([&](Int i) {
	Var<Vec3> c0 = MakeFloat3(1.0f, 0.0f, 0.0f);
	Var<Vec3> c1 = MakeFloat3(0.0f, 1.0f, 0.0f);
	Var<Vec3> c2 = MakeFloat3(0.0f, 0.0f, 1.0f);
	Var<Mat3> m	 = MakeMat3(c0, c1, c2);
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("mat3") != std::string::npos);
END_TEST

TEST(mat4_construction_codegen)
InspectorKernel1D inspector([&](Int i) {
	Var<Vec4> c0 = MakeFloat4(1.0f, 0.0f, 0.0f, 0.0f);
	Var<Vec4> c1 = MakeFloat4(0.0f, 1.0f, 0.0f, 0.0f);
	Var<Vec4> c2 = MakeFloat4(0.0f, 0.0f, 1.0f, 0.0f);
	Var<Vec4> c3 = MakeFloat4(0.0f, 0.0f, 0.0f, 1.0f);
	Var<Mat4> m	 = MakeMat4(c0, c1, c2, c3);
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("mat4") != std::string::npos);
END_TEST

TEST(mat4_from_cpu_constant)
Math::Mat4		  identity(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
InspectorKernel1D inspector([&](Int i) { Var<Mat4> m = MakeMat4(identity); });
std::string		  code = inspector.GetCode();
ASSERT(code.find("mat4(") != std::string::npos);
END_TEST

// =============================================================================
// Runtime: Mat4 Transform
// =============================================================================

TEST(mat4_identity_transform)
constexpr int			N = 4;
Math::Mat4				identity(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
std::vector<Math::Vec4> vecs = {
	Math::Vec4{1.0f, 2.0f, 3.0f, 1.0f},
	Math::Vec4{4.0f, 5.0f, 6.0f, 1.0f},
	Math::Vec4{0.0f, 0.0f, 0.0f, 1.0f},
	Math::Vec4{-1.0f, -2.0f, -3.0f, 1.0f},
};
Runtime::Buffer<Math::Vec4> bufVec(vecs);
Runtime::Buffer<Math::Vec4> bufOut(N);
Uniform<Math::Mat4>			uMat(identity);

Kernel1D					kernel(
	[&, N](Var<int> &id) {
		auto	  v	  = bufVec.Bind();
		auto	  out = bufOut.Bind();
		Var<Mat4> m	  = uMat.Load();
		out[id]		  = m * v[id];
	},
	256);

kernel.Dispatch(1, true);

std::vector<Math::Vec4> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i].x, vecs[i].x, 0.0001f);
	ASSERT_NEAR(output[i].y, vecs[i].y, 0.0001f);
	ASSERT_NEAR(output[i].z, vecs[i].z, 0.0001f);
	ASSERT_NEAR(output[i].w, vecs[i].w, 0.0001f);
}
END_TEST

TEST(mat4_scale_transform)
constexpr int			N = 3;
Math::Mat4				scaleMat(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1);
std::vector<Math::Vec4> vecs = {
	Math::Vec4{1.0f, 1.0f, 1.0f, 1.0f},
	Math::Vec4{2.0f, 3.0f, 4.0f, 1.0f},
	Math::Vec4{-1.0f, 0.5f, 2.0f, 1.0f},
};
Runtime::Buffer<Math::Vec4> bufVec(vecs);
Runtime::Buffer<Math::Vec4> bufOut(N);
Uniform<Math::Mat4>			uMat(scaleMat);

Kernel1D					kernel(
	[&, N](Var<int> &id) {
		auto	  v	  = bufVec.Bind();
		auto	  out = bufOut.Bind();
		Var<Mat4> m	  = uMat.Load();
		out[id]		  = m * v[id];
	},
	256);

kernel.Dispatch(1, true);

std::vector<Math::Vec4> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i].x, vecs[i].x * 2.0f, 0.0001f);
	ASSERT_NEAR(output[i].y, vecs[i].y * 3.0f, 0.0001f);
	ASSERT_NEAR(output[i].z, vecs[i].z * 4.0f, 0.0001f);
	ASSERT_NEAR(output[i].w, vecs[i].w, 0.0001f);
}
END_TEST

// =============================================================================
// Runtime: Matrix Addition / Subtraction
// =============================================================================

TEST(mat4_add_sub)
constexpr int				N = 2;
Math::Mat4					a(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
Math::Mat4					b(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
Runtime::Buffer<Math::Mat4> bufA(1);
Runtime::Buffer<Math::Mat4> bufB(1);
bufA.Upload(&a, 1);
bufB.Upload(&b, 1);
Runtime::Buffer<Math::Mat4> bufAdd(N);
Runtime::Buffer<Math::Mat4> bufSub(N);

Kernel1D					kernel(
	[&, N](Var<int> &id) {
		auto av	 = bufA.Bind();
		auto bv	 = bufB.Bind();
		auto add = bufAdd.Bind();
		auto sub = bufSub.Bind();
		add[id]	 = av[0] + bv[0];
		sub[id]	 = av[0] - bv[0];
	},
	256);

kernel.Dispatch(1, true);

std::vector<Math::Mat4> outAdd(1);
std::vector<Math::Mat4> outSub(1);
bufAdd.Download(outAdd.data(), 1);
bufSub.Download(outSub.data(), 1);

// Verify a few representative elements
ASSERT_NEAR(outAdd[0].m00, 2.0f, 0.0001f);
ASSERT_NEAR(outAdd[0].m11, 2.0f, 0.0001f);
ASSERT_NEAR(outAdd[0].m10, 1.0f, 0.0001f);
ASSERT_NEAR(outAdd[0].m01, 1.0f, 0.0001f);

ASSERT_NEAR(outSub[0].m00, 0.0f, 0.0001f);
ASSERT_NEAR(outSub[0].m11, 0.0f, 0.0001f);
ASSERT_NEAR(outSub[0].m10, -1.0f, 0.0001f);
ASSERT_NEAR(outSub[0].m01, -1.0f, 0.0001f);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Matrix Full Tests             " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_mat2_construction_codegen();
		test_mat3_construction_codegen();
		test_mat4_construction_codegen();
		test_mat4_from_cpu_constant();
		test_mat4_identity_transform();
		test_mat4_scale_transform();
		test_mat4_add_sub();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All matrix tests passed!               " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
