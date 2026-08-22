#include "ShaderOptimizationReport.h"

#include <spirv/unified1/spirv.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace GPU::Backend::Detail {

namespace {

constexpr uint32_t kSpirvHeaderWords		 = 5;
constexpr uint64_t kMaximumReportedDecisions = 4096;

struct InstructionView {
	size_t	 offset	   = 0;
	uint16_t opcode	   = 0;
	uint16_t wordCount = 0;
};

struct FunctionInfo {
	uint32_t	id = 0;
	std::string name;
	uint64_t	instructionCount	  = 0;
	uint64_t	branchCount			  = 0;
	uint64_t	textureOperationCount = 0;
	uint64_t	barrierCount		  = 0;
	uint64_t	callSiteCount		  = 0;
};

struct SourceLoop {
	size_t	 sourceOffset			  = 0;
	uint32_t line					  = 0;
	uint64_t tripCount				  = 0;
	uint64_t bodyInstructionEstimate  = 0;
	bool	 staticTripCountAvailable = false;
};

std::vector<InstructionView> Instructions(const std::vector<uint32_t> &spirv) {
	std::vector<InstructionView> result;
	if (spirv.size() < kSpirvHeaderWords) {
		return result;
	}
	for (size_t offset = kSpirvHeaderWords; offset < spirv.size();) {
		const uint32_t first	 = spirv[offset];
		const uint16_t wordCount = static_cast<uint16_t>(first >> 16u);
		const uint16_t opcode	 = static_cast<uint16_t>(first & 0xffffu);
		if (wordCount == 0 || offset + wordCount > spirv.size()) {
			result.clear();
			return result;
		}
		result.push_back({offset, opcode, wordCount});
		offset += wordCount;
	}
	return result;
}

bool IsBranch(uint16_t opcode) {
	return opcode == SpvOpBranch || opcode == SpvOpBranchConditional || opcode == SpvOpSwitch || opcode == SpvOpKill ||
		   opcode == SpvOpTerminateInvocation;
}

bool IsMemoryOperation(uint16_t opcode) {
	return opcode == SpvOpLoad || opcode == SpvOpStore || opcode == SpvOpCopyMemory || opcode == SpvOpCopyMemorySized ||
		   opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain || opcode == SpvOpPtrAccessChain ||
		   (opcode >= SpvOpAtomicLoad && opcode <= SpvOpAtomicXor);
}

bool IsTextureOperation(uint16_t opcode) {
	return opcode == SpvOpImageTexelPointer ||
		   (opcode >= SpvOpImageSampleImplicitLod && opcode <= SpvOpImageDrefGather) ||
		   (opcode >= SpvOpImageRead && opcode <= SpvOpImageQuerySamples);
}

bool IsBarrier(uint16_t opcode) {
	return opcode == SpvOpControlBarrier || opcode == SpvOpMemoryBarrier || opcode == SpvOpNamedBarrierInitialize ||
		   opcode == SpvOpMemoryNamedBarrier;
}

std::string DecodeString(const uint32_t *words, size_t wordCount) {
	std::string value;
	value.reserve(wordCount * sizeof(uint32_t));
	for (size_t index = 0; index < wordCount; ++index) {
		uint32_t word = words[index];
		for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
			const char character = static_cast<char>((word >> (byte * 8u)) & 0xffu);
			if (character == '\0') {
				return value;
			}
			value.push_back(character);
		}
	}
	return value;
}

std::string JsonEscape(std::string_view value) {
	std::ostringstream output;
	for (const unsigned char character : value) {
		switch (character) {
		case '\"':
			output << "\\\"";
			break;
		case '\\':
			output << "\\\\";
			break;
		case '\b':
			output << "\\b";
			break;
		case '\f':
			output << "\\f";
			break;
		case '\n':
			output << "\\n";
			break;
		case '\r':
			output << "\\r";
			break;
		case '\t':
			output << "\\t";
			break;
		default:
			if (character < 0x20u) {
				output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<uint32_t>(character)
					   << std::dec;
			} else {
				output << static_cast<char>(character);
			}
		}
	}
	return output.str();
}

std::string LevelName(ShaderOptimizationLevel level) {
	switch (level) {
	case ShaderOptimizationLevel::None:
		return "NONE";
	case ShaderOptimizationLevel::Size:
		return "SIZE";
	case ShaderOptimizationLevel::Aggressive:
		return "AGGRESSIVE";
	case ShaderOptimizationLevel::Ultra:
		return "ULTRA";
	case ShaderOptimizationLevel::Extreme:
		return "EXTREME";
	}
	return "UNKNOWN";
}

std::string MaskCommentsAndStrings(const std::string &source) {
	std::string masked = source;
	enum class State {
		Code,
		LineComment,
		BlockComment,
		String,
		Character
	};
	State state	  = State::Code;
	bool  escaped = false;
	for (size_t index = 0; index < source.size(); ++index) {
		const char current = source[index];
		const char next	   = index + 1 < source.size() ? source[index + 1] : '\0';
		switch (state) {
		case State::Code:
			if (current == '/' && next == '/') {
				masked[index] = masked[index + 1] = ' ';
				++index;
				state = State::LineComment;
			} else if (current == '/' && next == '*') {
				masked[index] = masked[index + 1] = ' ';
				++index;
				state = State::BlockComment;
			} else if (current == '\"') {
				masked[index] = ' ';
				state		  = State::String;
				escaped		  = false;
			} else if (current == '\'') {
				masked[index] = ' ';
				state		  = State::Character;
				escaped		  = false;
			}
			break;
		case State::LineComment:
			if (current == '\n') {
				state = State::Code;
			} else {
				masked[index] = ' ';
			}
			break;
		case State::BlockComment:
			if (current == '*' && next == '/') {
				masked[index] = masked[index + 1] = ' ';
				++index;
				state = State::Code;
			} else if (current != '\n') {
				masked[index] = ' ';
			}
			break;
		case State::String:
		case State::Character:
			if (current != '\n') {
				masked[index] = ' ';
			}
			if (!escaped &&
				((state == State::String && current == '\"') || (state == State::Character && current == '\''))) {
				state = State::Code;
			}
			escaped = !escaped && current == '\\';
			if (current != '\\')
				escaped = false;
			break;
		}
	}
	return masked;
}

bool ParseSigned(std::string_view text, int64_t &value) {
	const char *begin	  = text.data();
	const char *end		  = begin + text.size();
	auto [pointer, error] = std::from_chars(begin, end, value);
	return error == std::errc{} && pointer == end;
}

uint64_t SaturatingMultiply(uint64_t left, uint64_t right) {
	if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
		return std::numeric_limits<uint64_t>::max();
	return left * right;
}

uint64_t ComputeTripCount(int64_t start, int64_t bound, int64_t step, std::string_view comparison) {
	if (step == 0)
		return 0;
	if ((comparison == "<" || comparison == "<=") && step < 0)
		return 0;
	if ((comparison == ">" || comparison == ">=") && step > 0)
		return 0;
	if (comparison == "<" || comparison == "<=") {
		if (comparison == "<=" && bound == std::numeric_limits<int64_t>::max())
			return 0;
		const int64_t adjusted = comparison == "<=" ? bound + 1 : bound;
		if (start >= adjusted)
			return 0;
		const uint64_t distance = static_cast<uint64_t>(adjusted - start);
		return (distance + static_cast<uint64_t>(step) - 1u) / static_cast<uint64_t>(step);
	}
	if (comparison == ">=" && bound == std::numeric_limits<int64_t>::min())
		return 0;
	const int64_t adjusted = comparison == ">=" ? bound - 1 : bound;
	if (start <= adjusted)
		return 0;
	const uint64_t distance		= static_cast<uint64_t>(start - adjusted);
	const uint64_t positiveStep = static_cast<uint64_t>(-step);
	return (distance + positiveStep - 1u) / positiveStep;
}

uint64_t EstimateBodyInstructions(const std::string &masked, size_t afterHeader) {
	size_t begin = masked.find_first_not_of(" \t\r\n", afterHeader);
	if (begin == std::string::npos)
		return 1;
	size_t end = begin;
	if (masked[begin] == '{') {
		int depth = 1;
		for (end = begin + 1; end < masked.size() && depth > 0; ++end) {
			if (masked[end] == '{')
				++depth;
			else if (masked[end] == '}')
				--depth;
		}
	} else {
		end = masked.find(';', begin);
		if (end == std::string::npos)
			end = masked.size();
	}
	uint64_t estimate = 0;
	for (size_t index = begin; index < end; ++index) {
		if (masked[index] == ';')
			++estimate;
	}
	const std::string_view body(masked.data() + begin, end - begin);
	for (const std::string_view token : {"texture", "image", "barrier", "atomic", "if", "switch"}) {
		size_t position = 0;
		while ((position = body.find(token, position)) != std::string_view::npos) {
			++estimate;
			position += token.size();
		}
	}
	return std::max<uint64_t>(estimate, 1u);
}

std::vector<SourceLoop> ParseSourceLoops(const std::string &source) {
	const std::string masked = MaskCommentsAndStrings(source);
	const std::regex  header(
		R"(for\s*\(\s*(?:int|uint)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?[0-9]+)[uU]?\s*;\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*(-?[0-9]+)[uU]?\s*;\s*([^\)]*)\))");
	const std::regex		allFor(R"(for\s*\()");
	std::vector<SourceLoop> staticLoops;
	for (std::sregex_iterator iterator(masked.begin(), masked.end(), header), end; iterator != end; ++iterator) {
		const std::smatch &match = *iterator;
		SourceLoop		   loop;
		loop.sourceOffset = static_cast<size_t>(match.position());
		loop.line = static_cast<uint32_t>(1 + std::count(masked.begin(), masked.begin() + match.position(), '\n'));
		const std::string variable			= match[1].str();
		const std::string conditionVariable = match[3].str();
		int64_t			  start				= 0;
		int64_t			  bound				= 0;
		int64_t			  step				= 0;
		const std::string increment			= match[6].str();
		if (variable == conditionVariable && ParseSigned(match[2].str(), start) && ParseSigned(match[5].str(), bound)) {
			const std::regex incrementOne("(?:\\+\\+\\s*" + variable + "|" + variable + "\\s*\\+\\+|" + variable +
										  "\\s*\\+=\\s*1)");
			const std::regex decrementOne("(?:--\\s*" + variable + "|" + variable + "\\s*--|" + variable +
										  "\\s*-=\\s*1)");
			const std::regex addStep(variable + R"(\s*\+=\s*([0-9]+))");
			const std::regex subtractStep(variable + R"(\s*-=\s*([0-9]+))");
			std::smatch		 stepMatch;
			if (std::regex_match(increment, incrementOne))
				step = 1;
			else if (std::regex_match(increment, decrementOne))
				step = -1;
			else if (std::regex_match(increment, stepMatch, addStep) && ParseSigned(stepMatch[1].str(), step)) {
			} else if (std::regex_match(increment, stepMatch, subtractStep) && ParseSigned(stepMatch[1].str(), step))
				step = -step;
			if (step != 0) {
				loop.tripCount				  = ComputeTripCount(start, bound, step, match[4].str());
				loop.staticTripCountAvailable = loop.tripCount > 0;
			}
		}
		loop.bodyInstructionEstimate = EstimateBodyInstructions(masked, match.position() + match.length());
		staticLoops.push_back(loop);
	}

	std::vector<SourceLoop> result;
	for (std::sregex_iterator iterator(masked.begin(), masked.end(), allFor), end; iterator != end; ++iterator) {
		const size_t position = static_cast<size_t>(iterator->position());
		auto		 exact	  = std::find_if(staticLoops.begin(), staticLoops.end(),
											 [&](const SourceLoop &loop) { return loop.sourceOffset == position; });
		if (exact != staticLoops.end())
			result.push_back(*exact);
		else
			result.push_back({position,
							  static_cast<uint32_t>(1 + std::count(masked.begin(), masked.begin() + position, '\n')), 0,
							  1, false});
	}
	return result;
}

std::vector<size_t> LoopControlOffsets(const std::vector<uint32_t> &spirv) {
	std::vector<size_t> offsets;
	for (const InstructionView &instruction : Instructions(spirv)) {
		if (instruction.opcode == SpvOpLoopMerge && instruction.wordCount >= 4) {
			offsets.push_back(instruction.offset + 3);
		}
	}
	return offsets;
}

std::map<uint32_t, FunctionInfo> AnalyzeFunctions(const std::vector<uint32_t> &spirv) {
	std::unordered_map<uint32_t, std::string> names;
	for (const InstructionView &instruction : Instructions(spirv)) {
		if (instruction.opcode == SpvOpName && instruction.wordCount >= 3) {
			names[spirv[instruction.offset + 1]] =
				DecodeString(spirv.data() + instruction.offset + 2, instruction.wordCount - 2);
		}
	}

	std::map<uint32_t, FunctionInfo> functions;
	FunctionInfo					*current = nullptr;
	for (const InstructionView &instruction : Instructions(spirv)) {
		if (instruction.opcode == SpvOpFunction && instruction.wordCount >= 5) {
			const uint32_t id		  = spirv[instruction.offset + 2];
			auto [iterator, inserted] = functions.emplace(id, FunctionInfo{});
			(void)inserted;
			current		  = &iterator->second;
			current->id	  = id;
			current->name = names.contains(id) ? names[id] : ("function_%" + std::to_string(id));
		} else if (instruction.opcode == SpvOpFunctionEnd) {
			current = nullptr;
		} else if (current != nullptr) {
			++current->instructionCount;
			if (IsBranch(instruction.opcode))
				++current->branchCount;
			if (IsTextureOperation(instruction.opcode))
				++current->textureOperationCount;
			if (IsBarrier(instruction.opcode))
				++current->barrierCount;
		}
		if (instruction.opcode == SpvOpFunctionCall && instruction.wordCount >= 4) {
			const uint32_t target	  = spirv[instruction.offset + 3];
			auto [iterator, inserted] = functions.emplace(target, FunctionInfo{});
			if (inserted) {
				iterator->second.id = target;
				iterator->second.name =
					names.contains(target) ? names[target] : ("function_%" + std::to_string(target));
			}
			++iterator->second.callSiteCount;
		}
	}
	return functions;
}

void LimitsForLevel(ShaderOptimizationLevel level, uint64_t &tripCount, uint64_t &codeGrowth, uint64_t &valuePressure,
					uint64_t &inlineInstructions) {
	switch (level) {
	case ShaderOptimizationLevel::None:
		tripCount = codeGrowth = valuePressure = inlineInstructions = 0;
		break;
	case ShaderOptimizationLevel::Size:
		tripCount		   = 4;
		codeGrowth		   = 64;
		valuePressure	   = 16;
		inlineInstructions = 12;
		break;
	case ShaderOptimizationLevel::Aggressive:
		tripCount		   = 8;
		codeGrowth		   = 192;
		valuePressure	   = 32;
		inlineInstructions = 32;
		break;
	case ShaderOptimizationLevel::Ultra:
		tripCount		   = 16;
		codeGrowth		   = 384;
		valuePressure	   = 40;
		inlineInstructions = 48;
		break;
	case ShaderOptimizationLevel::Extreme:
		tripCount		   = 32;
		codeGrowth		   = 768;
		valuePressure	   = 48;
		inlineInstructions = 64;
		break;
	}
}

void AppendAvailability(std::ostringstream &output, bool available, std::string_view reasonCode,
						const std::string &valueJson) {
	output << "{\"available\":" << (available ? "true" : "false");
	if (!available)
		output << ",\"reasonCode\":\"" << JsonEscape(reasonCode) << "\"";
	if (available)
		output << ",\"value\":" << valueJson;
	output << '}';
}

void AppendMetrics(std::ostringstream &output, const OptimizationMetrics &metrics) {
	output << "{\"binaryBytes\":\"" << metrics.binaryBytes << "\",\"instructionCount\":\"" << metrics.instructionCount
		   << "\",\"functionCount\":\"" << metrics.functionCount << "\",\"functionCallCount\":\""
		   << metrics.functionCallCount << "\",\"loopCount\":\"" << metrics.loopCount << "\",\"branchCount\":\""
		   << metrics.branchCount << "\",\"memoryOperationCount\":\"" << metrics.memoryOperationCount
		   << "\",\"textureOperationCount\":\"" << metrics.textureOperationCount << "\",\"barrierCount\":\""
		   << metrics.barrierCount << "\",\"maximumFunctionInstructionCount\":\""
		   << metrics.maximumFunctionInstructionCount
		   << "\",\"staticValuePressureUpperBound\":{\"available\":true,\"unit\":\"SPIRV_FUNCTION_INSTRUCTIONS\","
			  "\"value\":\""
		   << metrics.staticValuePressureUpperBound << "\"},\"workGroupSize\":";
	if (metrics.workGroupSizeAvailable) {
		output << "{\"available\":true,\"x\":" << metrics.workGroupSizeX << ",\"y\":" << metrics.workGroupSizeY
			   << ",\"z\":" << metrics.workGroupSizeZ << '}';
	} else {
		output << "{\"available\":false,\"reasonCode\":\"NOT_A_COMPUTE_LOCAL_SIZE\"}";
	}
	output << '}';
}

} // namespace

OptimizationMetrics AnalyzeOptimizationMetrics(const std::vector<uint32_t> &spirv) {
	OptimizationMetrics metrics;
	metrics.binaryBytes					 = spirv.size() * sizeof(uint32_t);
	uint64_t currentFunctionInstructions = 0;
	bool	 insideFunction				 = false;
	for (const InstructionView &instruction : Instructions(spirv)) {
		++metrics.instructionCount;
		if (instruction.opcode == SpvOpFunction) {
			insideFunction				= true;
			currentFunctionInstructions = 0;
			++metrics.functionCount;
		} else if (instruction.opcode == SpvOpFunctionEnd) {
			insideFunction = false;
			metrics.maximumFunctionInstructionCount =
				std::max(metrics.maximumFunctionInstructionCount, currentFunctionInstructions);
		} else if (insideFunction) {
			++currentFunctionInstructions;
		}
		if (instruction.opcode == SpvOpFunctionCall)
			++metrics.functionCallCount;
		if (instruction.opcode == SpvOpLoopMerge)
			++metrics.loopCount;
		if (IsBranch(instruction.opcode))
			++metrics.branchCount;
		if (IsMemoryOperation(instruction.opcode))
			++metrics.memoryOperationCount;
		if (IsTextureOperation(instruction.opcode))
			++metrics.textureOperationCount;
		if (IsBarrier(instruction.opcode))
			++metrics.barrierCount;
		if (instruction.opcode == SpvOpExecutionMode && instruction.wordCount >= 6 &&
			spirv[instruction.offset + 2] == SpvExecutionModeLocalSize) {
			metrics.workGroupSizeX		   = spirv[instruction.offset + 3];
			metrics.workGroupSizeY		   = spirv[instruction.offset + 4];
			metrics.workGroupSizeZ		   = spirv[instruction.offset + 5];
			metrics.workGroupSizeAvailable = true;
		}
	}
	metrics.staticValuePressureUpperBound = metrics.maximumFunctionInstructionCount;
	return metrics;
}

OptimizationPlan BuildOptimizationPlan(const std::string &source, const std::vector<uint32_t> &spirv,
									   ShaderOptimizationLevel level, const OptimizationTargetFacts &target) {
	OptimizationPlan plan;
	plan.level						= level;
	plan.target						= target;
	plan.before						= AnalyzeOptimizationMetrics(spirv);

	uint64_t tripLimit				= 0;
	uint64_t growthLimit			= 0;
	uint64_t pressureLimit			= 0;
	uint64_t inlineInstructionLimit = 0;
	LimitsForLevel(level, tripLimit, growthLimit, pressureLimit, inlineInstructionLimit);
	const uint64_t workGroupInvocations = static_cast<uint64_t>(plan.before.workGroupSizeX) *
										  static_cast<uint64_t>(plan.before.workGroupSizeY) *
										  static_cast<uint64_t>(plan.before.workGroupSizeZ);
	if (plan.before.workGroupSizeAvailable && workGroupInvocations >= 256) {
		pressureLimit = pressureLimit == 0 ? 0 : std::max<uint64_t>(pressureLimit * 3u / 4u, 8u);
	}

	const std::vector<size_t>	  loopOffsets	   = LoopControlOffsets(spirv);
	const std::vector<SourceLoop> sourceLoops	   = ParseSourceLoops(source);
	const size_t				  loopCount		   = std::max(loopOffsets.size(), sourceLoops.size());
	const bool					  exactLoopMapping = loopOffsets.size() == sourceLoops.size();
	for (size_t index = 0; index < loopCount && plan.decisions.size() < kMaximumReportedDecisions; ++index) {
		OptimizationDecision decision;
		decision.decisionId			  = "unroll-" + std::to_string(index + 1);
		decision.kind				  = "UNROLL";
		decision.chosenAction		  = "KEEP_ROLLED";
		decision.tripCountLimit		  = tripLimit;
		decision.codeGrowthLimitBytes = growthLimit;
		decision.valuePressureLimit	  = pressureLimit;
		if (index < sourceLoops.size()) {
			decision.sourceLine				  = sourceLoops[index].line;
			decision.sourceLineAvailable	  = true;
			decision.staticTripCount		  = sourceLoops[index].tripCount;
			decision.staticTripCountAvailable = sourceLoops[index].staticTripCountAvailable;
			decision.instructionEstimate	  = sourceLoops[index].bodyInstructionEstimate;
			decision.estimatedCodeGrowthBytes =
				sourceLoops[index].staticTripCountAvailable
					? SaturatingMultiply(
						  SaturatingMultiply(sourceLoops[index].tripCount, sourceLoops[index].bodyInstructionEstimate),
						  sizeof(uint32_t))
					: 0;
			decision.estimatedValuePressureUnits = sourceLoops[index].bodyInstructionEstimate;
		}
		if (index < loopOffsets.size()) {
			decision.inputIrWordOffset			= loopOffsets[index] - 3;
			decision.inputIrWordOffsetAvailable = true;
		}

		if (level == ShaderOptimizationLevel::None) {
			decision.status		= "NOT_APPLICABLE";
			decision.reasonCode = "OPTIMIZATION_PROFILE_DISABLED";
		} else if (!exactLoopMapping) {
			decision.status		= "REJECTED_LEGALITY";
			decision.reasonCode = "LOOP_SOURCE_IR_COUNT_MISMATCH";
			if (index < loopOffsets.size())
				plan.rejectedLoopControlWordOffsets.push_back(loopOffsets[index]);
		} else if (index >= loopOffsets.size()) {
			decision.status		= "REJECTED_LEGALITY";
			decision.reasonCode = "SOURCE_LOOP_HAS_NO_SPIRV_LOOP";
		} else if (!decision.staticTripCountAvailable) {
			decision.status		= "NOT_APPLICABLE";
			decision.reasonCode = "UNROLL_STATIC_TRIP_COUNT_REQUIRED";
			plan.rejectedLoopControlWordOffsets.push_back(loopOffsets[index]);
		} else if (decision.staticTripCount > tripLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "UNROLL_TRIP_COUNT_LIMIT";
			plan.rejectedLoopControlWordOffsets.push_back(loopOffsets[index]);
		} else if (decision.estimatedCodeGrowthBytes > growthLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "UNROLL_CODE_SIZE_LIMIT";
			plan.rejectedLoopControlWordOffsets.push_back(loopOffsets[index]);
		} else if (decision.estimatedValuePressureUnits > pressureLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "UNROLL_VALUE_PRESSURE_LIMIT";
			plan.rejectedLoopControlWordOffsets.push_back(loopOffsets[index]);
		} else {
			decision.status		  = "APPLIED";
			decision.reasonCode	  = "UNROLL_WITHIN_TARGET_COST_LIMITS";
			decision.chosenAction = "REQUEST_FULL_UNROLL";
			decision.requested	  = true;
			plan.acceptedLoopControlWordOffsets.push_back(loopOffsets[index]);
		}
		plan.decisions.push_back(std::move(decision));
	}
	if (loopCount == 0) {
		plan.decisions.push_back({"unroll-none", "UNROLL", "NOT_APPLICABLE", "NO_LOOP_CANDIDATES", "NO_ACTION"});
	}

	bool anyInlineCandidate = false;
	bool anyInlineRejected	= false;
	for (const auto &[functionId, function] : AnalyzeFunctions(spirv)) {
		if (function.callSiteCount == 0 || plan.decisions.size() >= kMaximumReportedDecisions)
			continue;
		anyInlineCandidate = true;
		OptimizationDecision decision;
		decision.decisionId			 = "inline-" + std::to_string(functionId);
		decision.kind				 = "INLINE";
		decision.targetFunctionId	 = functionId;
		decision.targetFunctionName	 = function.name;
		decision.callSiteCount		 = function.callSiteCount;
		decision.instructionEstimate = function.instructionCount;
		decision.estimatedCodeGrowthBytes =
			SaturatingMultiply(SaturatingMultiply(function.instructionCount, function.callSiteCount), sizeof(uint32_t));
		decision.estimatedValuePressureUnits = function.instructionCount;
		decision.codeGrowthLimitBytes		 = growthLimit;
		decision.valuePressureLimit			 = pressureLimit;
		decision.tripCountLimit				 = inlineInstructionLimit;
		decision.chosenAction				 = "KEEP_CALL";
		if (level == ShaderOptimizationLevel::None) {
			decision.status		= "NOT_APPLICABLE";
			decision.reasonCode = "OPTIMIZATION_PROFILE_DISABLED";
		} else if (function.barrierCount > 0) {
			decision.status		= "REJECTED_LEGALITY";
			decision.reasonCode = "INLINE_BARRIER_CALLEE_REJECTED";
			anyInlineRejected	= true;
		} else if (function.instructionCount > inlineInstructionLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "INLINE_INSTRUCTION_LIMIT";
			anyInlineRejected	= true;
		} else if (decision.estimatedCodeGrowthBytes > growthLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "INLINE_CODE_SIZE_LIMIT";
			anyInlineRejected	= true;
		} else if (decision.estimatedValuePressureUnits > pressureLimit) {
			decision.status		= "REJECTED_COST";
			decision.reasonCode = "INLINE_VALUE_PRESSURE_LIMIT";
			anyInlineRejected	= true;
		} else {
			decision.status		  = "APPLIED";
			decision.reasonCode	  = "INLINE_WITHIN_TARGET_COST_LIMITS";
			decision.chosenAction = "REQUEST_INLINE";
			decision.requested	  = true;
		}
		plan.decisions.push_back(std::move(decision));
	}
	plan.allowExhaustiveInline = !anyInlineRejected;
	if (anyInlineRejected) {
		for (OptimizationDecision &decision : plan.decisions) {
			if (decision.kind == "INLINE" && decision.requested) {
				decision.requested	  = false;
				decision.status		  = "REJECTED_COST";
				decision.reasonCode	  = "INLINE_MODULE_CONSERVATIVE_GUARD";
				decision.chosenAction = "KEEP_CALL";
			}
		}
	}
	if (!anyInlineCandidate) {
		plan.decisions.push_back(
			{"inline-none", "INLINE", "NOT_APPLICABLE", "NO_FUNCTION_CALL_CANDIDATES", "NO_ACTION"});
	}

	plan.decisions.push_back(
		{"vectorize-module", "VECTORIZE", "NOT_APPLICABLE", "VECTORIZE_COST_MODEL_NOT_CONFIGURED", "NO_ACTION"});
	plan.decisions.push_back(
		{"specialize-module", "SPECIALIZE", "NOT_APPLICABLE", "NO_SPECIALIZATION_CANDIDATES", "NO_ACTION"});
	OptimizationDecision barrier{"barrier-module", "BARRIER", "NOT_APPLICABLE", "NO_BARRIER_CANDIDATES", "NO_ACTION"};
	if (plan.before.barrierCount > 0) {
		barrier.status		 = "REJECTED_LEGALITY";
		barrier.reasonCode	 = "BARRIER_REMOVAL_REQUIRES_RESOURCE_SCHEDULE_PROOF";
		barrier.chosenAction = "PRESERVE_BARRIERS";
	}
	plan.decisions.push_back(std::move(barrier));
	return plan;
}

void ApplyOptimizationPlan(std::vector<uint32_t> &spirv, const OptimizationPlan &plan) {
	for (const size_t offset : plan.acceptedLoopControlWordOffsets) {
		if (offset >= spirv.size())
			continue;
		spirv[offset] &= ~static_cast<uint32_t>(SpvLoopControlDontUnrollMask);
		spirv[offset] |= static_cast<uint32_t>(SpvLoopControlUnrollMask);
	}
	for (const size_t offset : plan.rejectedLoopControlWordOffsets) {
		if (offset >= spirv.size())
			continue;
		spirv[offset] &= ~static_cast<uint32_t>(SpvLoopControlUnrollMask);
		spirv[offset] |= static_cast<uint32_t>(SpvLoopControlDontUnrollMask);
	}
}

void FinalizeOptimizationPlan(OptimizationPlan &plan, const std::vector<uint32_t> &optimized) {
	const OptimizationMetrics after			 = AnalyzeOptimizationMetrics(optimized);
	const uint64_t			  requestedLoops = static_cast<uint64_t>(
		std::count_if(plan.decisions.begin(), plan.decisions.end(), [](const OptimizationDecision &decision) {
			return decision.requested && decision.kind == "UNROLL";
		}));
	const uint64_t removedLoops = plan.before.loopCount > after.loopCount ? plan.before.loopCount - after.loopCount : 0;
	uint64_t	   requestedCallSites = 0;
	for (const OptimizationDecision &decision : plan.decisions) {
		if (decision.requested && decision.kind == "INLINE")
			requestedCallSites += decision.callSiteCount;
	}
	const uint64_t removedCallSites = plan.before.functionCallCount > after.functionCallCount
										  ? plan.before.functionCallCount - after.functionCallCount
										  : 0;
	for (OptimizationDecision &decision : plan.decisions) {
		if (!decision.requested)
			continue;
		if (decision.kind == "UNROLL" && removedLoops < requestedLoops) {
			decision.status		  = "REJECTED_LEGALITY";
			decision.reasonCode	  = "SPIRV_UNROLL_LEGALITY_REJECTED";
			decision.chosenAction = "KEEP_ROLLED";
		} else if (decision.kind == "INLINE" && removedCallSites < requestedCallSites) {
			decision.status		  = "REJECTED_LEGALITY";
			decision.reasonCode	  = "SPIRV_INLINE_LEGALITY_REJECTED";
			decision.chosenAction = "KEEP_CALL";
		}
	}
}

std::string SerializeOptimizationReport(const OptimizationPlan &plan, const OptimizationMetrics &after,
										const std::string &optimizerVersion, const std::string &frontendVersion) {
	std::ostringstream output;
	output << "{\"schemaVersion\":1,\"kind\":\"EasyGPU.ShaderOptimizationReport\",\"optimizer\":{"
		   << "\"name\":\"SPIRV-Tools with EasyGPU target cost gates\",\"version\":\"" << JsonEscape(optimizerVersion)
		   << "\",\"frontendVersion\":\"" << JsonEscape(frontendVersion)
		   << "\",\"costModelVersion\":\"easygpu-vulkan-cost-v1\",\"profile\":\"" << LevelName(plan.level)
		   << "\"},\"target\":{\"backend\":\"VULKAN\",\"deviceName\":\"" << JsonEscape(plan.target.deviceName)
		   << "\",\"vendorId\":" << plan.target.vendorId << ",\"deviceId\":" << plan.target.deviceId
		   << ",\"driverVersion\":" << plan.target.driverVersion << ",\"apiVersion\":" << plan.target.apiVersion
		   << ",\"maxComputeWorkGroupInvocations\":" << plan.target.maxComputeWorkGroupInvocations
		   << ",\"maxComputeSharedMemoryBytes\":" << plan.target.maxComputeSharedMemoryBytes
		   << ",\"maxPerStageResources\":" << plan.target.maxPerStageResources
		   << ",\"registerAllocation\":{\"available\":false,\"reasonCode\":\"DRIVER_REGISTER_ALLOCATION_NOT_EXPOSED\"}"
		   << ",\"occupancy\":{\"available\":false,\"reasonCode\":\"DRIVER_OCCUPANCY_NOT_EXPOSED\"}},"
		   << "\"metrics\":{\"before\":";
	AppendMetrics(output, plan.before);
	output << ",\"after\":";
	AppendMetrics(output, after);
	const int64_t codeSizeDelta =
		static_cast<int64_t>(after.binaryBytes) - static_cast<int64_t>(plan.before.binaryBytes);
	const int64_t pressureDelta = static_cast<int64_t>(after.staticValuePressureUpperBound) -
								  static_cast<int64_t>(plan.before.staticValuePressureUpperBound);
	output << ",\"actualEffects\":{\"scope\":\"WHOLE_SHADER\",\"codeSizeDeltaBytes\":" << codeSizeDelta
		   << ",\"staticValuePressureDelta\":{\"available\":true,\"unit\":\"SPIRV_FUNCTION_INSTRUCTIONS\",\"value\":"
		   << pressureDelta
		   << "},\"registerDelta\":{\"available\":false,\"reasonCode\":\"DRIVER_REGISTER_ALLOCATION_NOT_EXPOSED\"}"
		   << ",\"occupancyDelta\":{\"available\":false,\"reasonCode\":\"DRIVER_OCCUPANCY_NOT_EXPOSED\"}}},"
		   << "\"decisions\":[";
	for (size_t index = 0; index < plan.decisions.size(); ++index) {
		if (index != 0)
			output << ',';
		const OptimizationDecision &decision = plan.decisions[index];
		output << "{\"decisionId\":\"" << JsonEscape(decision.decisionId) << "\",\"kind\":\""
			   << JsonEscape(decision.kind) << "\",\"status\":\"" << JsonEscape(decision.status)
			   << "\",\"reasonCode\":\"" << JsonEscape(decision.reasonCode) << "\",\"chosenAction\":\""
			   << JsonEscape(decision.chosenAction) << "\",\"costInputs\":{\"staticTripCount\":";
		AppendAvailability(output, decision.staticTripCountAvailable, "STATIC_TRIP_COUNT_UNAVAILABLE",
						   "\"" + std::to_string(decision.staticTripCount) + "\"");
		output
			<< ",\"instructionEstimate\":\"" << decision.instructionEstimate << "\",\"callSiteCount\":\""
			<< decision.callSiteCount << "\",\"estimatedCodeGrowthBytes\":\"" << decision.estimatedCodeGrowthBytes
			<< "\",\"estimatedValuePressure\":{\"available\":true,\"unit\":\"SPIRV_FUNCTION_INSTRUCTIONS\",\"value\":\""
			<< decision.estimatedValuePressureUnits
			<< "\"},\"divergentControlFlow\":{\"available\":false,\"reasonCode\":\"DIVERGENCE_ANALYSIS_NOT_EXPOSED\"}"
			<< ",\"sharedMemoryBytes\":{\"available\":false,\"reasonCode\":\"STATIC_SHARED_MEMORY_ANALYSIS_NOT_"
			   "EXPOSED\"}"
			<< ",\"textureMemoryTraffic\":{\"available\":true,\"unit\":\"STATIC_SPIRV_OPERATIONS\",\"value\":\""
			<< plan.before.textureOperationCount + plan.before.memoryOperationCount << "\"}},"
			<< "\"limits\":{\"tripCount\":\"" << decision.tripCountLimit << "\",\"codeGrowthBytes\":\""
			<< decision.codeGrowthLimitBytes << "\",\"valuePressure\":\"" << decision.valuePressureLimit << "\"},"
			<< "\"estimatedEffects\":{\"benefit\":{\"available\":true,\"unit\":\"ELIMINATED_CONTROL_OPERATIONS\","
			   "\"value\":\""
			<< (decision.requested ? std::max<uint64_t>(decision.callSiteCount, 1u) : 0u)
			<< "\"},\"codeSizeDeltaBytes\":\"" << decision.estimatedCodeGrowthBytes
			<< "\",\"registerDelta\":{\"available\":false,\"reasonCode\":\"DRIVER_REGISTER_ALLOCATION_NOT_EXPOSED\"}"
			<< ",\"occupancyDelta\":{\"available\":false,\"reasonCode\":\"DRIVER_OCCUPANCY_NOT_EXPOSED\"}},"
			<< "\"sourceLocation\":";
		if (decision.sourceLineAvailable) {
			output << "{\"available\":true,\"coordinateSpace\":\"BACKEND_INPUT_GLSL\",\"line\":" << decision.sourceLine
				   << '}';
		} else {
			output << "{\"available\":false,\"reasonCode\":\"SOURCE_LOCATION_NOT_EXPOSED\"}";
		}
		output << ",\"irLocation\":";
		if (decision.inputIrWordOffsetAvailable) {
			output << "{\"available\":true,\"coordinateSpace\":\"PRE_OPT_SPIRV_WORD_OFFSET\",\"wordOffset\":\""
				   << decision.inputIrWordOffset << "\"}";
		} else {
			output << "{\"available\":false,\"reasonCode\":\"IR_LOCATION_NOT_EXPOSED\"}";
		}
		if (decision.targetFunctionId != 0) {
			output << ",\"targetFunction\":{\"spirvId\":" << decision.targetFunctionId << ",\"name\":\""
				   << JsonEscape(decision.targetFunctionName) << "\"}";
		}
		output << '}';
	}
	output << "]}";
	return output.str();
}

} // namespace GPU::Backend::Detail
