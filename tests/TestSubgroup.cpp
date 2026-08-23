#include <Backend/Backend.h>
#include <IR/Module.h>
#include <Kernel/KernelBuildContext.h>
#include <Runtime/Context.h>

#include <iostream>
#include <stdexcept>
#include <string>

#define REQUIRE(condition)                                                                                             \
	do {                                                                                                               \
		if (!(condition))                                                                                              \
			throw std::runtime_error("Requirement failed: " #condition);                                               \
	} while (false)

namespace {

std::string buildOrdinaryShader() {
	GPU::IR::ModuleBuilder builder;
	builder.BeginComputeKernel(1, 1, 1, 1);
	const auto one = builder.Literal(GPU::IR::Type::UInt(), "1u");
	builder.Expression(one);
	auto context = GPU::IR::BuildKernelBuildContext(builder.GetModule());
	REQUIRE(context != nullptr);
	return context->GetCompleteCode();
}

std::string buildSubgroupShader() {
	GPU::IR::ModuleBuilder builder;
	builder.BeginComputeKernel(48, 1, 1, 1);

	const auto one			= builder.Literal(GPU::IR::Type::UInt(), "1u");
	const auto zero			= builder.Literal(GPU::IR::Type::UInt(), "0u");
	const auto lane			= builder.SubgroupInvocationId();
	const auto parity		= builder.Binary(GPU::IR::BinaryOp::BitAnd, lane, one);
	const auto predicate	= builder.Compare(GPU::IR::CompareOp::Equal, parity, zero);
	const auto always		= builder.Literal(GPU::IR::Type::Bool(), "true");
	const auto activeMask	= builder.SubgroupBallot(always);
	const auto trueMask		= builder.SubgroupBallot(predicate);
	const auto activeCount	= builder.SubgroupBallotBitCount(activeMask);
	const auto trueCount	= builder.SubgroupBallotBitCount(trueMask);
	const auto anyTrue		= builder.SubgroupAny(predicate);
	const auto allTrue		= builder.SubgroupAll(predicate);
	const auto elected		= builder.SubgroupElect();
	const auto subgroupId	= builder.SubgroupId();
	const auto subgroupSize = builder.SubgroupSize();
	const auto numSubgroups = builder.NumSubgroups();

	REQUIRE(lane != GPU::IR::InvalidValueId);
	REQUIRE(activeMask != GPU::IR::InvalidValueId);
	REQUIRE(trueMask != GPU::IR::InvalidValueId);
	REQUIRE(activeCount != GPU::IR::InvalidValueId);
	REQUIRE(trueCount != GPU::IR::InvalidValueId);
	REQUIRE(anyTrue != GPU::IR::InvalidValueId);
	REQUIRE(allTrue != GPU::IR::InvalidValueId);
	REQUIRE(elected != GPU::IR::InvalidValueId);
	REQUIRE(subgroupId != GPU::IR::InvalidValueId);
	REQUIRE(subgroupSize != GPU::IR::InvalidValueId);
	REQUIRE(numSubgroups != GPU::IR::InvalidValueId);

	builder.DeclareLocal(GPU::IR::Type::UInt4(), "active_mask", activeMask);
	builder.DeclareLocal(GPU::IR::Type::UInt4(), "true_mask", trueMask);
	builder.DeclareLocal(GPU::IR::Type::UInt(), "active_count", activeCount);
	builder.DeclareLocal(GPU::IR::Type::UInt(), "true_count", trueCount);
	builder.DeclareLocal(GPU::IR::Type::Bool(), "any_true", anyTrue);
	builder.DeclareLocal(GPU::IR::Type::Bool(), "all_true", allTrue);
	builder.DeclareLocal(GPU::IR::Type::Bool(), "elected", elected);
	builder.DeclareLocal(GPU::IR::Type::UInt(), "subgroup_id", subgroupId);
	builder.DeclareLocal(GPU::IR::Type::UInt(), "subgroup_size", subgroupSize);
	builder.DeclareLocal(GPU::IR::Type::UInt(), "num_subgroups", numSubgroups);

	const auto &module = builder.GetModule();
	REQUIRE(module.subgroupRequirements.basic);
	REQUIRE(module.subgroupRequirements.vote);
	REQUIRE(module.subgroupRequirements.ballot);
	auto context = GPU::IR::BuildKernelBuildContext(module);
	REQUIRE(context != nullptr);
	REQUIRE(context->RequiresSubgroupBasic());
	REQUIRE(context->RequiresSubgroupVote());
	REQUIRE(context->RequiresSubgroupBallot());
	return context->GetCompleteCode();
}

} // namespace

int main() {
	const std::string ordinary = buildOrdinaryShader();
	REQUIRE(ordinary.find("GL_KHR_shader_subgroup") == std::string::npos);
	REQUIRE(ordinary.find("subgroupBallot") == std::string::npos);

	const std::string subgroup = buildSubgroupShader();
	REQUIRE(subgroup.find("#extension GL_KHR_shader_subgroup_basic : require") != std::string::npos);
	REQUIRE(subgroup.find("#extension GL_KHR_shader_subgroup_vote : require") != std::string::npos);
	REQUIRE(subgroup.find("#extension GL_KHR_shader_subgroup_ballot : require") != std::string::npos);
	REQUIRE(subgroup.find("subgroupBallot(true)") != std::string::npos);
	REQUIRE(subgroup.find("subgroupBallotBitCount") != std::string::npos);
	REQUIRE(subgroup.find("subgroupAny") != std::string::npos);
	REQUIRE(subgroup.find("subgroupAll") != std::string::npos);
	REQUIRE(subgroup.find("subgroupElect") != std::string::npos);
	REQUIRE(subgroup.find("gl_SubgroupInvocationID") != std::string::npos);
	REQUIRE(subgroup.find("gl_SubgroupID") != std::string::npos);
	REQUIRE(subgroup.find("gl_SubgroupSize") != std::string::npos);
	REQUIRE(subgroup.find("gl_NumSubgroups") != std::string::npos);

	GPU::Runtime::AutoInitContext();
	auto *backend = GPU::Runtime::Context::GetBackend();
	REQUIRE(backend != nullptr);
	const auto &caps = backend->GetCaps();
	if (caps.supportsComputeSubgroups || caps.supportsSubgroupBasic || caps.supportsSubgroupVote ||
		caps.supportsSubgroupBallot) {
		REQUIRE(caps.subgroupSize > 0);
	}

	if (caps.supportsComputeSubgroups && caps.supportsSubgroupBasic && caps.supportsSubgroupVote &&
		caps.supportsSubgroupBallot) {
		GPU::Backend::ShaderDesc descriptor;
		descriptor.type		  = GPU::Backend::ShaderType::Compute;
		descriptor.sourceCode = subgroup;
		descriptor.entryPoint = "main";
		const auto shader	  = backend->CreateShader(descriptor);
		REQUIRE(shader != GPU::Backend::INVALID_SHADER_HANDLE);
		backend->DestroyShader(shader);
	}

	std::cout << "Subgroup IR, capability, and shader compilation tests passed.\n";
	return 0;
}
