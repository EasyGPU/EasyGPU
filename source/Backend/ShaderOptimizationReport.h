#pragma once

#include <Backend/Backend.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace GPU::Backend::Detail {

struct OptimizationTargetFacts {
	std::string deviceName;
	uint32_t	vendorId					   = 0;
	uint32_t	deviceId					   = 0;
	uint32_t	driverVersion				   = 0;
	uint32_t	apiVersion					   = 0;
	uint32_t	maxComputeWorkGroupInvocations = 0;
	uint32_t	maxComputeSharedMemoryBytes	   = 0;
	uint32_t	maxPerStageResources		   = 0;
};

struct OptimizationMetrics {
	uint64_t binaryBytes					 = 0;
	uint64_t instructionCount				 = 0;
	uint64_t functionCount					 = 0;
	uint64_t functionCallCount				 = 0;
	uint64_t loopCount						 = 0;
	uint64_t branchCount					 = 0;
	uint64_t memoryOperationCount			 = 0;
	uint64_t textureOperationCount			 = 0;
	uint64_t barrierCount					 = 0;
	uint64_t maximumFunctionInstructionCount = 0;
	uint64_t staticValuePressureUpperBound	 = 0;
	uint32_t workGroupSizeX					 = 1;
	uint32_t workGroupSizeY					 = 1;
	uint32_t workGroupSizeZ					 = 1;
	bool	 workGroupSizeAvailable			 = false;
};

struct OptimizationDecision {
	std::string decisionId;
	std::string kind;
	std::string status;
	std::string reasonCode;
	std::string chosenAction;
	uint32_t	sourceLine					= 0;
	bool		sourceLineAvailable			= false;
	size_t		inputIrWordOffset			= 0;
	bool		inputIrWordOffsetAvailable	= false;
	uint64_t	staticTripCount				= 0;
	bool		staticTripCountAvailable	= false;
	uint64_t	instructionEstimate			= 0;
	uint64_t	callSiteCount				= 0;
	uint64_t	estimatedCodeGrowthBytes	= 0;
	uint64_t	estimatedValuePressureUnits = 0;
	uint64_t	tripCountLimit				= 0;
	uint64_t	codeGrowthLimitBytes		= 0;
	uint64_t	valuePressureLimit			= 0;
	uint32_t	targetFunctionId			= 0;
	std::string targetFunctionName;
	bool		requested = false;
};

struct OptimizationPlan {
	ShaderOptimizationLevel			  level = ShaderOptimizationLevel::None;
	OptimizationTargetFacts			  target;
	OptimizationMetrics				  before;
	std::vector<OptimizationDecision> decisions;
	std::vector<size_t>				  acceptedLoopControlWordOffsets;
	std::vector<size_t>				  rejectedLoopControlWordOffsets;
	bool							  allowExhaustiveInline = true;
};

OptimizationPlan	BuildOptimizationPlan(const std::string &source, const std::vector<uint32_t> &spirv,
										  ShaderOptimizationLevel level, const OptimizationTargetFacts &target);
void				ApplyOptimizationPlan(std::vector<uint32_t> &spirv, const OptimizationPlan &plan);
OptimizationMetrics AnalyzeOptimizationMetrics(const std::vector<uint32_t> &spirv);
void				FinalizeOptimizationPlan(OptimizationPlan &plan, const std::vector<uint32_t> &optimized);
std::string SerializeOptimizationReport(const OptimizationPlan &plan, const OptimizationMetrics &after,
										const std::string &optimizerVersion, const std::string &frontendVersion);

} // namespace GPU::Backend::Detail
