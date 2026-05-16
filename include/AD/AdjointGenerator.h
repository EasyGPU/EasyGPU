#pragma once

/**
 * @file AdjointGenerator.h
 * @brief Generates adjoint (gradient) GLSL code by walking the tape in reverse.
 *
 * The AdjointGenerator takes a recorded GradientTape and produces the complete
 * GLSL source for the backward pass. It walks the tape entries in reverse order
 * and applies the corresponding gradient rules to generate adjoint accumulation
 * statements.
 */

#ifndef EASYGPU_AD_ADJOINTGENERATOR_H
#define EASYGPU_AD_ADJOINTGENERATOR_H

#include <AD/AdjointTable.h>
#include <AD/GradientTape.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace GPU::AD {

/**
 * The parts of a generated adjoint pass, for merging into a forward shader.
 */
struct AdjointBody {
	/** (adjointName, glslType) pairs for variable declarations. */
	std::vector<std::pair<std::string, std::string>> declarations;
	/** Adjoint accumulation statements (body lines). */
	std::vector<std::string> lines;
	/** (paramName, adjName) pairs for gradient write-back. */
	std::vector<std::pair<std::string, std::string>> writebacks;
	/** GLSL definitions of adjoint functions for Callable bodies. */
	std::string callableAdjointFunctions;
};

/**
 * Generates the backward-pass GLSL code from a recorded gradient tape.
 *
 * Usage:
 *   AdjointGenerator gen;
 *   std::string glsl = gen.Generate(tape);
 *
 * The generated GLSL contains:
 *   1. Adjoint variable declarations (initialized to zero)
 *   2. Seed: adjoint of loss = 1.0
 *   3. Reverse-order gradient accumulation statements
 *   4. Parameter gradient write-back to output buffers
 */
class AdjointGenerator {
public:
	AdjointGenerator();

	/**
	 * Generate the complete backward-pass GLSL code from a tape.
	 * @param tape The recorded gradient tape from the forward pass
	 * @param writeBackParams If true, emit code to write parameter adjoints to output buffers
	 * @return Complete GLSL source for the backward pass
	 */
	std::string Generate(const GradientTape &tape, bool writeBackParams = true);

		/**
		 * Generate only the adjoint body parts (no main() wrapper).
		 * Returns declarations, body lines, and writeback pairs for merging
		 * into an existing forward shader's main().
		 */
		AdjointBody GenerateBody(const GradientTape &tape, bool writeBackParams = true);

	/**
	 * Get the adjoint table after generation (for querying adjoint variable names).
	 */
	const AdjointTable &GetAdjointTable() const { return _adjTable; }

private:
	// ---- Entry processing -------------------------------------------------

	void ProcessEntry(const TapeEntry &entry);
	void ProcessBinaryOp(const TapeEntry &entry);
	void ProcessUnaryOp(const TapeEntry &entry);
	void ProcessIntrinsic1(const TapeEntry &entry);
	void ProcessIntrinsic2(const TapeEntry &entry);
	void ProcessIntrinsic3(const TapeEntry &entry);
	void ProcessTernary(const TapeEntry &entry);
	void ProcessCompoundAssign(const TapeEntry &entry);
	void ProcessControlFlowBegin(const TapeEntry &entry);
	void ProcessControlFlowEnd();
	void ProcessCall(const TapeEntry &entry);
	void ProcessReturn(const TapeEntry &entry);

	// ---- Control flow helpers ---------------------------------------------

	/** Add a generated line to the current collector frame (or top-level body). */
	void EmitLine(const std::string &line);

	/** Push a new control flow collector frame. */
	void PushControlFrame();

	/** Pop the top frame, wrap its contents, and emit to the parent frame. */
	void PopControlFrameAndWrap(const TapeEntry &beginEntry);

	// ---- Gradient rule helpers --------------------------------------------

	/** Emit: d_input += expression; */
	void EmitAccumulate(const std::string &inputName, const std::string &gradExpr);

	/** Build the adjoint variable name for a forward variable. */
	std::string Adj(const std::string &varName);

	/** Make a zero literal of the given GLSL type. */
	static std::string ZeroOf(const std::string &glslType);

	// ---- Intrinsic gradient rules -----------------------------------------

	using GradientRule = std::function<void(AdjointGenerator			   *gen,
											const TapeEntry			   &entry,
											const std::string		   &dOut,
											const std::vector<std::string> &inputs)>;

	void RegisterIntrinsicRules();
	void RegisterArithmeticRules();

	// ---- Output -----------------------------------------------------------

	/** Final assembly: declarations + body + writeback */
	std::string Assemble();

	// ---- State ------------------------------------------------------------

	AdjointTable _adjTable;

	// Generated GLSL lines for the backward pass body
	std::vector<std::string> _bodyLines;

	// Control flow collector stack (for nesting if/for)
	struct ControlFrame {
		std::vector<std::string> lines;			// Current collected adjoint lines
		std::vector<std::string> chainBlocks;	// For if-chains: accumulated wrapped branches
		bool					 isIfChain = false;
	};
	std::vector<ControlFrame> _controlStack;

	/** Get the currently active line buffer (top of stack or bodyLines). */
	std::vector<std::string> &ActiveLines();

	// Intrinsic function name -> gradient rule
	std::unordered_map<std::string, GradientRule> _intrinsicRules;

	// OperationCode -> gradient rule (for binary/unary ops)
	std::unordered_map<int, GradientRule> _arithmeticRules;

	// Parameter names that need gradient write-back
	std::vector<std::pair<std::string, std::string>> _paramWritebacks;

	// Reference to the main tape (for sub-tape access during Call processing)
	const GradientTape *_tape = nullptr;

	// Pre-built map: entry id → call index (for O(1) sub-tape lookup)
	std::unordered_map<int32_t, int> _callIndexMap;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADJOINTGENERATOR_H
