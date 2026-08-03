/**
 * @file TestBuilderHardening.cpp
 * @brief Regression tests for Builder/DSL state hardening.
 */

#include <AD/GradientTape.h>
#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>
#include <Kernel/KernelBuildContext.h>
#include <Runtime/Buffer.h>

#include <cassert>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Builder;
using namespace GPU::IR::Value;

template <typename T>
concept HasExprModulo = requires(Expr<T> a, Expr<T> b) { a % b; };

class DummyContext : public BuilderContext {
public:
	void PushTranslatedCode(std::string Code) override {
		code += std::move(Code);
	}

	std::string AssignVarName() override {
		return "v" + std::to_string(++nextVar);
	}

	bool HasStructDefinition(const std::string &) const override {
		return false;
	}

	void AddStructDefinition(const std::string &, const std::string &) override {
	}

	const std::vector<std::string> &GetStructDefinitions() const override {
		return emptyStrings;
	}

	uint32_t AllocateBindingSlot() override {
		return nextBinding++;
	}

	void RegisterBuffer(uint32_t, const std::string &, const std::string &, int) override {
	}

	std::string GetBufferDeclarations() const override {
		return "";
	}

	const std::vector<uint32_t> &GetBufferBindings() const override {
		return emptyBindings;
	}

	void BindRuntimeBuffer(uint32_t, Backend::BufferHandle) override {
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const override {
		return emptyRuntimeBindings;
	}

	uint32_t AllocateTextureBinding() override {
		return nextTextureBinding++;
	}

	void RegisterTexture(uint32_t, Runtime::PixelFormat, const std::string &, uint32_t, uint32_t, bool) override {
	}

	std::string GetTextureDeclarations() const override {
		return "";
	}

	const std::vector<uint32_t> &GetTextureBindings() const override {
		return emptyBindings;
	}

	void BindRuntimeTexture(uint32_t, uint32_t) override {
	}

	void BindRuntimeTextureSampler(uint32_t, const Backend::SamplerDesc &) override {
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const override {
		return emptyRuntimeBindings;
	}

	const std::unordered_map<uint32_t, Backend::SamplerDesc> &GetRuntimeTextureSamplerBindings() const override {
		return emptyRuntimeTextureSamplers;
	}

	std::string RegisterUniform(const std::string &, void *, size_t, size_t,
								std::function<void(uint32_t, const std::string &, void *)>,
								std::function<void(void *, void *)>) override {
		return "u_dummy";
	}

	std::string GetUniformDeclarations() const override {
		return "";
	}

	void AddCallableDeclaration(const std::string &) override {
	}

	void AddCallableBodyGenerator(std::function<void()>) override {
	}

	void PushCallableBody() override {
	}

	void PopCallableBody() override {
	}

	std::vector<std::string> GetCallableDeclarations() const override {
		return {};
	}

	std::string GenerateCallableBodies() override {
		return "";
	}

	std::string code;
	int			nextVar			   = 0;
	uint32_t	nextBinding		   = 0;
	uint32_t	nextTextureBinding = 0;

private:
	static inline const std::vector<std::string>			   emptyStrings;
	static inline const std::vector<uint32_t>				   emptyBindings;
	static inline const std::unordered_map<uint32_t, uint32_t> emptyRuntimeBindings;
	static inline const std::unordered_map<uint32_t, Backend::SamplerDesc> emptyRuntimeTextureSamplers;
};

static void test_scoped_bind_restores_previous_context() {
	auto		&builder = Builder::Get();
	DummyContext outer;
	DummyContext inner;

	{
		Builder::ScopedBind outerBind(builder, outer);
		assert(builder.Context() == &outer);
		{
			Builder::ScopedBind innerBind(builder, inner);
			assert(builder.Context() == &inner);
		}
		assert(builder.Context() == &outer);
	}
	assert(builder.Context() == nullptr);
}

static void test_runtime_texture_sampler_binding_is_cached() {
	GPU::Kernel::KernelBuildContext context(1);
	context.RegisterTexture(0, Runtime::PixelFormat::RGBA8, "texture", 2, 2, true);
	context.BindRuntimeTexture(0, 1);

	Backend::SamplerDesc sampler;
	sampler.minFilter = Backend::SamplerFilter::Linear;
	sampler.magFilter = Backend::SamplerFilter::Linear;
	context.BindRuntimeTextureSampler(0, sampler);

	const auto &bindings = context.GetCachedBindings();
	assert(bindings.size() == 1);
	assert(bindings[0].type == Backend::BindingType::Sampler);
	assert(bindings[0].samplerOverridden);
	assert(bindings[0].sampler.minFilter == Backend::SamplerFilter::Linear);
	assert(bindings[0].sampler.magFilter == Backend::SamplerFilter::Linear);
}

static void test_scoped_gradient_tape_restores_previous_state() {
	auto &builder = Builder::Get();

	assert(builder.GetGradientTape() == nullptr);
	{
		AD::GradientTape			dummyTape;
		Builder::ScopedGradientTape guard(builder, &dummyTape);
		assert(builder.GetGradientTape() == &dummyTape);
	}
	assert(builder.GetGradientTape() == nullptr);
}

static void test_var_const_rvalue_semantics_removed() {
	static_assert(std::is_move_constructible_v<Var<int>>);
	static_assert(!std::is_constructible_v<Var<int>, const Var<int> &&>);
	static_assert(!std::is_assignable_v<Var<int> &, const Var<int> &&>);
}

static void test_expr_modulo_type_constraints() {
	static_assert(HasExprModulo<int>);
	static_assert(!HasExprModulo<float>);
}

static void test_buffer_type_names() {
	assert(Runtime::GetGLSLTypeNameForBuffer<float>() == "float");
	assert(Runtime::GetGLSLTypeNameForBuffer<Math::Vec3>() == "vec3");
	assert(Runtime::GetGLSLTypeNameForBuffer<Math::Mat4>() == "mat4");
}

static void test_barrier_flags_are_contiguous() {
	static_assert(static_cast<uint32_t>(Backend::BarrierType::Buffer) == 1u);
	static_assert(static_cast<uint32_t>(Backend::BarrierType::Texture) == 2u);
	static_assert(static_cast<uint32_t>(Backend::BarrierType::Uniform) == 4u);
	static_assert(static_cast<uint32_t>(Backend::BarrierType::All) == 7u);
}

static void test_unbind_without_context_throws() {
	auto &builder = Builder::Get();
	assert(builder.Context() == nullptr);
	bool caught = false;
	try {
		builder.Unbind();
	} catch (const std::runtime_error &) {
		caught = true;
	}
	assert(caught);
}

int main() {
	std::cout << "========================================\n";
	std::cout << "  EasyGPU Builder Hardening Tests       \n";
	std::cout << "========================================\n";

	test_scoped_bind_restores_previous_context();
	test_runtime_texture_sampler_binding_is_cached();
	test_scoped_gradient_tape_restores_previous_state();
	test_var_const_rvalue_semantics_removed();
	test_expr_modulo_type_constraints();
	test_buffer_type_names();
	test_barrier_flags_are_contiguous();
	test_unbind_without_context_throws();

	std::cout << "All builder hardening tests passed.\n";
	return 0;
}
