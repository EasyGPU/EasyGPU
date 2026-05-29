/**
 * @file AdjointGenerator.cpp
 * @brief Implementation of the backward-pass adjoint code generator.
 *
 * Walks the gradient tape in reverse order and applies gradient rules for
 * each recorded operation. Produces GLSL source code that computes gradients
 * of the scalar loss with respect to all active variables.
 */

#include <AD/AdjointGenerator.h>

#include <AD/TapeEntry.h>

#include <cstdio>
#include <format>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <algorithm>

namespace GPU::AD {

// =============================================================================
// Constructor & Rule Registration
// =============================================================================

AdjointGenerator::AdjointGenerator() : _tmpCounter(0) {
	RegisterArithmeticRules();
	RegisterIntrinsicRules();
}

void AdjointGenerator::RegisterArithmeticRules() {
	using Op = GPU::IR::Node::OperationCode;

	// --- Add: z = a + b  →  d_a += d_z;  d_b += d_z ---
	_arithmeticRules[static_cast<int>(Op::Add)] = [](AdjointGenerator *gen, const TapeEntry &e,
													  const std::string			&dOut,
													  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], dOut);
		gen->EmitAccumulate(in[1], dOut);
	};

	// --- Sub: z = a - b  →  d_a += d_z;  d_b += -d_z ---
	_arithmeticRules[static_cast<int>(Op::Sub)] = [](AdjointGenerator *gen, const TapeEntry &e,
													  const std::string			&dOut,
													  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], dOut);
		gen->EmitAccumulate(in[1], std::format("-({})", dOut));
	};

	// --- Mul: z = a * b  →  d_a += d_z * b;  d_b += d_z * a ---
	_arithmeticRules[static_cast<int>(Op::Mul)] = [](AdjointGenerator *gen, const TapeEntry &e,
													  const std::string			&dOut,
													  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*({})", dOut, in[1]));
		gen->EmitAccumulate(in[1], std::format("({})*({})", dOut, in[0]));
	};

	// --- Div: z = a / b  →  d_a += d_z / b;  d_b += -d_z * a / (b*b) ---
	_arithmeticRules[static_cast<int>(Op::Div)] = [](AdjointGenerator *gen, const TapeEntry &e,
													  const std::string			&dOut,
													  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/({})", dOut, in[1]));
		gen->EmitAccumulate(in[1], std::format("-(({})*({}))/(({})*({}))", dOut, in[0], in[1], in[1]));
	};

	// --- Neg: z = -a  →  d_a += -d_z ---
	_arithmeticRules[static_cast<int>(Op::Neg)] = [](AdjointGenerator *gen, const TapeEntry &e,
													  const std::string			&dOut,
													  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("-({})", dOut));
	};
}

void AdjointGenerator::RegisterIntrinsicRules() {
	// =========================================================================
	// Single-parameter intrinsics
	// =========================================================================

	// sin(x)  →  d_x += d_out * cos(x)
	_intrinsicRules["sin"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*cos({})", dOut, in[0]));
	};

	// cos(x)  →  d_x += d_out * (-sin(x))
	_intrinsicRules["cos"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(-sin({}))", dOut, in[0]));
	};

	// exp(x)  →  d_x += d_out * exp(x)
	_intrinsicRules["exp"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*exp({})", dOut, in[0]));
	};

	// log(x)  →  d_x += d_out / x
	_intrinsicRules["log"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/({})", dOut, in[0]));
	};

	// sqrt(x)  →  d_x += d_out / (2.0 * sqrt(x))
	_intrinsicRules["sqrt"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/(2.0*sqrt({}))", dOut, in[0]));
	};

	// abs(x)  →  d_x += d_out * sign(x)
	_intrinsicRules["abs"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*sign({})", dOut, in[0]));
	};

	// tan(x)  →  d_x += d_out * (1.0 + tan(x)*tan(x))
	_intrinsicRules["tan"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(1.0+tan({})*tan({}))", dOut, in[0], in[0]));
	};

	// asin(x)  →  d_x += d_out / sqrt(1.0 - x*x)
	_intrinsicRules["asin"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/sqrt(1.0-({})*({}))", dOut, in[0], in[0]));
	};

	// acos(x)  →  d_x += d_out * (-1.0 / sqrt(1.0 - x*x))
	_intrinsicRules["acos"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(-1.0/sqrt(1.0-({})*({})))", dOut, in[0], in[0]));
	};

	// atan(x)  →  d_x += d_out / (1.0 + x*x)
	// atan2(y, x)  →  d_y += d_out * x / (x*x + y*y);  d_x += d_out * (-y) / (x*x + y*y)
	// GLSL uses 'atan' for both; dispatch on argument count.
	_intrinsicRules["atan"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		if (in.size() >= 2) {
			std::string denom = std::format("({})*({})+({})*({})", in[0], in[0], in[1], in[1]);
			gen->EmitAccumulate(in[0], std::format("({})*({})/({})", dOut, in[1], denom));
			gen->EmitAccumulate(in[1], std::format("({})*(-({}))/({})", dOut, in[0], denom));
		} else {
			gen->EmitAccumulate(in[0], std::format("({})/(1.0+({})*({}))", dOut, in[0], in[0]));
		}
	};

	// sinh(x)  →  d_x += d_out * cosh(x)
	_intrinsicRules["sinh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*cosh({})", dOut, in[0]));
	};

	// cosh(x)  →  d_x += d_out * sinh(x)
	_intrinsicRules["cosh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*sinh({})", dOut, in[0]));
	};

	// tanh(x)  →  d_x += d_out * (1.0 - tanh(x)*tanh(x))
	_intrinsicRules["tanh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(1.0-tanh({})*tanh({}))", dOut, in[0], in[0]));
	};

	// exp2(x)  →  d_x += d_out * log(2.0) * exp2(x)
	_intrinsicRules["exp2"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*log(2.0)*exp2({})", dOut, in[0]));
	};

	// log2(x)  →  d_x += d_out / (x * log(2.0))
	_intrinsicRules["log2"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								 const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/(({})*log(2.0))", dOut, in[0]));
	};

	// inversesqrt(x)  →  d_x += -0.5 * d_out / (x * sqrt(x))
	_intrinsicRules["inversesqrt"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
										const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("-0.5*({})/(({})*sqrt({}))", dOut, in[0], in[0]));
	};

	// fract(x)  →  d_x += d_out  (gradient of fractional part is 1)
	_intrinsicRules["fract"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], dOut);
	};

	// radians(x)  →  d_x += d_out * (PI / 180.0)
	_intrinsicRules["radians"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*0.01745329252", dOut));
	};

	// degrees(x)  →  d_x += d_out * (180.0 / PI)
	_intrinsicRules["degrees"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*57.295779513", dOut));
	};

	// asinh(x)  →  d_x += d_out / sqrt(x^2 + 1)
	_intrinsicRules["asinh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/sqrt(({})*({})+1.0)", dOut, in[0], in[0]));
	};

	// acosh(x)  →  d_x += d_out / sqrt(x^2 - 1)
	_intrinsicRules["acosh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/sqrt(({})*({})-1.0)", dOut, in[0], in[0]));
	};

	// atanh(x)  →  d_x += d_out / (1 - x^2)
	_intrinsicRules["atanh"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})/(1.0-({})*({}))", dOut, in[0], in[0]));
	};

	// Non-differentiable single-param functions (floor, ceil, trunc, round, sign)
	auto zeroGrad1 = [](AdjointGenerator *, const TapeEntry &, const std::string &,
						const std::vector<std::string> &) { /* gradient is 0 */ };
	_intrinsicRules["floor"] = zeroGrad1;
	_intrinsicRules["ceil"] = zeroGrad1;
	_intrinsicRules["trunc"] = zeroGrad1;
	_intrinsicRules["round"] = zeroGrad1;
	_intrinsicRules["roundEven"] = zeroGrad1;
	_intrinsicRules["sign"] = zeroGrad1;

	// =========================================================================
	// Two-parameter intrinsics
	// =========================================================================

	// pow(a, b)  →  d_a += d_out * b * pow(a, b-1);  d_b += d_out * pow(a, b) * log(a)
	_intrinsicRules["pow"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0],
							std::format("({})*({})*pow({},({})-1.0)", dOut, in[1], in[0], in[1]));
		gen->EmitAccumulate(in[1],
							std::format("({})*pow({},{})*log({})", dOut, in[0], in[1], in[0]));
	};

	// min(a, b)  →  subgradient: pass to whichever is smaller
	_intrinsicRules["min"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*step({},{})", dOut, in[0], in[1]));
		gen->EmitAccumulate(in[1], std::format("({})*(1.0-step({},{}))", dOut, in[0], in[1]));
	};

	// max(a, b)  →  subgradient: pass to whichever is larger
	_intrinsicRules["max"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(1.0-step({},{}))", dOut, in[0], in[1]));
		gen->EmitAccumulate(in[1], std::format("({})*step({},{})", dOut, in[0], in[1]));
	};

	// mod(x, y)  →  d_x += d_out;  d_y += -d_out * trunc(x/y)
	_intrinsicRules["mod"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], dOut);
		gen->EmitAccumulate(in[1], std::format("-({})*trunc(({})/({}))", dOut, in[0], in[1]));
	};

	// step(edge, x)  →  d_x += 0 (discontinuous), d_edge += 0
	_intrinsicRules["step"] = zeroGrad1;

	// =========================================================================
	// Three-parameter intrinsics
	// =========================================================================

	// clamp(x, lo, hi)  →  piecewise gradient
	_intrinsicRules["clamp"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*step({},{})*(1.0-step({},{}))", dOut, in[1], in[0],
											   in[2], in[0]));
		gen->EmitAccumulate(in[1], std::format("({})*(1.0-step({},{}))", dOut, in[1], in[0]));
		gen->EmitAccumulate(in[2], std::format("({})*step({},{})", dOut, in[2], in[0]));
	};

	// mix(a, b, t)  →  d_a += d_out * (1-t);  d_b += d_out * t;  d_t += d_out * (b-a)
	_intrinsicRules["mix"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*(1.0-({}))", dOut, in[2]));
		gen->EmitAccumulate(in[1], std::format("({})*({})", dOut, in[2]));
		gen->EmitAccumulate(in[2], std::format("({})*(({})-({}))", dOut, in[1], in[0]));
	};

	// smoothstep(e0, e1, x)  →  Hermite derivatives w.r.t. all three inputs
	// t = clamp((x-e0)/(e1-e0), 0, 1); deriv = 6*t*(1-t)
	// dx  = d_out * deriv / (e1-e0)
	// de0 = -d_out * deriv * (e1-x) / (e1-e0)^2
	// de1 = -d_out * deriv * (x-e0) / (e1-e0)^2
	_intrinsicRules["smoothstep"] = [](AdjointGenerator *gen, const TapeEntry &e,
									   const std::string			  &dOut,
									   const std::vector<std::string> &in) {
		const auto &e0 = in[0], &e1 = in[1], &x = in[2];
		std::string t  = std::format("clamp((({})-({}))/(({})-({})),0.0,1.0)", x, e0, e1, e0);
		std::string df = std::format("6.0*{}*(1.0-{})", t, t);
		std::string den = std::format("({})-({})", e1, e0);
		gen->EmitAccumulate(x,  std::format("({})*{}/({})", dOut, df, den));
		gen->EmitAccumulate(e0, std::format("-({})*{}*(({})-({}))/(({})*({}))", dOut, df, e1, x, den, den));
		gen->EmitAccumulate(e1, std::format("-({})*{}*(({})-({}))/(({})*({}))", dOut, df, x, e0, den, den));
	};

	// =========================================================================
	// Geometric functions (vector ops)
	// =========================================================================

	// dot(a, b)  →  d_a += d_out * b;  d_b += d_out * a
	_intrinsicRules["dot"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*({})", dOut, in[1]));
		gen->EmitAccumulate(in[1], std::format("({})*({})", dOut, in[0]));
	};

	// cross(a, b)  →  d_a += cross(b, d_out);  d_b += cross(d_out, a)
	_intrinsicRules["cross"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								  const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("cross({},{})", in[1], dOut));
		gen->EmitAccumulate(in[1], std::format("cross({},{})", dOut, in[0]));
	};

	// length(x)  →  d_x += d_out * (x / length(x))
	_intrinsicRules["length"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
								   const std::vector<std::string> &in) {
		gen->EmitAccumulate(in[0], std::format("({})*({})/length({})", dOut, in[0], in[0]));
	};

	// normalize(x)  →  d_x += (d_out - x_n * dot(x_n, d_out)) / length(x)
	// where x_n = x / length(x)
	_intrinsicRules["normalize"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									  const std::vector<std::string> &in) {
		std::string xn = std::format("({})/length({})", in[0], in[0]);
		std::string proj = std::format("({})*dot({},{})", xn, xn, dOut);
		std::string tangent = std::format("({})-({})", dOut, proj);
		gen->EmitAccumulate(in[0], std::format("({})/length({})", tangent, in[0]));
	};

	// distance(p0, p1)  →  d_p0 += d_out * (p0-p1)/len;  d_p1 += d_out * (p1-p0)/len
	_intrinsicRules["distance"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									 const std::vector<std::string> &in) {
		std::string diff = std::format("({})-({})", in[0], in[1]);
		gen->EmitAccumulate(in[0], std::format("({})*({})/length({})", dOut, diff, diff));
		gen->EmitAccumulate(in[1], std::format("({})*(({})-({}))/length({})", dOut, in[1], in[0], diff));
	};

	// reflect(I, N)  →  d_I += d_out;  d_N += -2*dot(N,I)*d_out - 2*dot(d_out,I)*N
	// Simplified: treat as not differentiable w.r.t. N (normal is usually fixed)
	_intrinsicRules["reflect"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									const std::vector<std::string> &in) {
		std::string dotNI = std::format("dot({},{})", in[1], in[0]);
		gen->EmitAccumulate(in[0],
			std::format("({})-2.0*({})*({})", dOut, in[1], dotNI));
	};

	// refract(I, N, eta)  →  approximate gradient for I
	_intrinsicRules["refract"] = [](AdjointGenerator *gen, const TapeEntry &e, const std::string &dOut,
									const std::vector<std::string> &in) {
		if (in.size() >= 3) {
			gen->EmitAccumulate(in[0], dOut);
		}
	};

	// faceforward(N, I, Nref) — discrete sign-based, zero gradient
	_intrinsicRules["faceforward"] = [](AdjointGenerator *, const TapeEntry &, const std::string &,
										const std::vector<std::string> &) { /* zero gradient */ };

	// Non-differentiable on integer params
	_intrinsicRules["floatBitsToInt"] = zeroGrad1;
	_intrinsicRules["floatBitsToUint"] = zeroGrad1;
	_intrinsicRules["intBitsToFloat"] = zeroGrad1;
	_intrinsicRules["uintBitsToFloat"] = zeroGrad1;
}

// =============================================================================
// Main Generation Entry Point
// =============================================================================

std::string AdjointGenerator::Generate(const GradientTape &tape, bool writeBackParams) {
	_tape = &tape;
	_adjTable.Clear();
	_bodyLines.clear();
	_paramWritebacks.clear();

	if (tape.Size() == 0) return "";

	// Step 0: Build the active set by backward propagation from loss/parameters.
	// MarkLoss() is called after the forward computation, so the tape's
	// real-time active propagation doesn't reach intermediate variables.
	// We walk the tape in reverse to find every variable that contributes
	// to the loss or to a registered parameter.
	std::unordered_set<std::string> activeSet;
	if (tape.LossVar()) {
		activeSet.insert(tape.LossVar()->name);
	}
	for (const auto &[paramName, paramType] : tape.Parameters()) {
		activeSet.insert(paramName);
	}

	// Filter: skip literal/constant names (GLSL constructors, booleans).
	auto isLiteral = [](const std::string &name) {
		if (name.empty()) return true;
		// Boolean literals
		if (name == "true" || name == "false") return true;
		// GLSL type constructors: float(...), vec2(...), etc.
		// Buffer accesses look like buf[...] and are NOT literals.
		// Only treat as literal if it starts with a GLSL type name followed by '('.
		static const char* glslTypes[] = {
			"float","int","uint","bool",
			"vec2","vec3","vec4",
			"ivec2","ivec3","ivec4",
			"uvec2","uvec3","uvec4",
			"bvec2","bvec3","bvec4",
			"mat2","mat3","mat4",
			"dvec2","dvec3","dvec4","dmat2","dmat3","dmat4"
		};
		for (const char* t : glslTypes) {
			size_t tlen = std::char_traits<char>::length(t);
			if (name.compare(0, tlen, t) == 0 && name.size() > tlen && name[tlen] == '(') {
				return true;
			}
		}
		return false;
	};

	// Extract base buffer name from a variable name.
	// For "buf1[expr]" -> "buf1". For "v123" -> returns "" (not a buffer).
	auto bufBaseName = [](const std::string &name) -> std::string {
		auto bpos = name.find('[');
		if (bpos == std::string::npos) return "";
		return name.substr(0, bpos);
	};

	// Helper: check if a name is active, matching both full name and
	// buffer base name (so buf1[expr1] matches buf1[expr2]).
	auto isActive = [&](const std::string &name) -> bool {
		if (activeSet.count(name)) return true;
		std::string base = bufBaseName(name);
		return !base.empty() && activeSet.count(base);
	};

	// Iterate reverse until the active set stabilizes.
	bool changed = true;
	int passNum = 0;
	while (changed) {
		changed = false;
		passNum++;
		int addedThisPass = 0;
		for (int32_t i = static_cast<int32_t>(tape.Size()) - 1; i >= 0; --i) {
			const auto &entry = tape[i];
			if (isActive(entry.output.name)) {
				for (const auto &in : entry.inputs) {
					if (isLiteral(in.name)) continue;
					if (activeSet.insert(in.name).second) {
						// Also add base buffer name for producer matching
						std::string base = bufBaseName(in.name);
						if (!base.empty()) activeSet.insert(base);
						changed = true;
						addedThisPass++;
					}
				}
			}
		}

	}

	// Build transitive alias map for resolving stale aliases in loops
	BuildAliasMap();

	// Step 1: Pre-allocate adjoint variables for all active variables.
	for (const auto &entry : tape.Entries()) {
		if (!isActive(entry.output.name)) continue;

		_adjTable.GetOrCreate(entry.output.name, entry.output.glslType);
		for (const auto &in : entry.inputs) {
			if (isActive(in.name)) {
				_adjTable.GetOrCreate(in.name, in.glslType);
			}
		}
	}

	// Also ensure adjoints exist for registered parameters
	for (const auto &[paramName, paramType] : tape.Parameters()) {
		_adjTable.GetOrCreate(paramName, paramType);
	}

	// Count how many registered parameters belong to each buffer base.
	{
		std::unordered_map<std::string, size_t> bufParamCount;
		for (const auto &[paramName2, paramType2] : tape.Parameters()) {
			auto bpos = paramName2.find('[');
			if (bpos != std::string::npos) {
				bufParamCount[paramName2.substr(0, bpos)]++;
			}
		}
		for (const auto &[bufBase, count] : bufParamCount) {
			std::string adjName = AdjointTable::MakeAdjointName(bufBase + "[0]");
			if (!adjName.empty()) {
				_adjTable.SetArraySize(adjName, count);
			}
		}
	}

	// Step 2: Seed the backward pass: adjoint of loss = 1.0
	if (tape.LossVar()) {
		const auto &lossVar = *tape.LossVar();
		std::string adjLoss = _adjTable.GetOrCreate(lossVar.name, lossVar.glslType);
		EmitLine(std::format("{} = {}(1.0);", adjLoss, lossVar.glslType));
	}

	// Step 2.5: Build call index map for O(1) sub-tape lookup
	_callIndexMap.clear();
	{
		int callIdx = 0;
		for (int32_t i = 0; i < static_cast<int32_t>(tape.Size()); i++) {
			if (tape[i].kind == TapeOpKind::Call) {
				_callIndexMap[tape[i].id] = callIdx++;
			}
		}
	}
	//}

	// Step 3: Walk tape in reverse, generating adjoint statements
	for (int32_t i = static_cast<int32_t>(tape.Size()) - 1; i >= 0; --i) {
		ProcessEntry(tape[i]);
	}

	// Step 4: Post-process — resolve alias variables in generated lines
	if (!_aliasMap.empty()) {
		for (auto &line : _bodyLines) {
			line = ResolveAliases(line);
		}
	}

	// Step 5: Collect parameter write-backs
	if (writeBackParams) {
		for (const auto &[paramName, paramType] : tape.Parameters()) {
			std::string adjName = _adjTable.Get(paramName);
			if (!adjName.empty()) {
				_paramWritebacks.emplace_back(ResolveAliases(paramName),
					ResolveAliases(adjName));
			}
		}
	}

	// Step 6: Assemble final GLSL
	return Assemble();
}

AdjointBody AdjointGenerator::GenerateBody(const GradientTape &tape, bool writeBackParams) {
	Generate(tape, writeBackParams);
	AdjointBody body;
	body.declarations = _adjTable.AllDeclarations();
	body.lines = std::move(_bodyLines);
	body.writebacks = std::move(_paramWritebacks);
	return body;
}

// =============================================================================
// Entry Processing
// =============================================================================

void AdjointGenerator::ProcessEntry(const TapeEntry &entry) {
	switch (entry.kind) {
	case TapeOpKind::BinaryOp:
		ProcessBinaryOp(entry);
		break;
	case TapeOpKind::UnaryOp:
		ProcessUnaryOp(entry);
		break;
	case TapeOpKind::ExpressionGradient:
		ProcessExpressionGradient(entry);
		break;
	case TapeOpKind::Intrinsic1:
		ProcessIntrinsic1(entry);
		break;
	case TapeOpKind::Intrinsic2:
		ProcessIntrinsic2(entry);
		break;
	case TapeOpKind::Intrinsic3:
		ProcessIntrinsic3(entry);
		break;
	case TapeOpKind::Ternary:
		ProcessTernary(entry);
		break;
	case TapeOpKind::CompoundAssign:
		ProcessCompoundAssign(entry);
		break;
	case TapeOpKind::ControlFlowBegin:
		ProcessControlFlowBegin(entry);
		break;
	case TapeOpKind::ControlFlowEnd:
		ProcessControlFlowEnd();
		break;
	case TapeOpKind::Call:
		ProcessCall(entry);
		break;
	case TapeOpKind::Return:
		ProcessReturn(entry);
		break;
	default:
		break;
	}
}

void AdjointGenerator::ProcessBinaryOp(const TapeEntry &entry) {
	std::string dOut = _adjTable.Get(entry.output.name);
	if (dOut.empty()) return;

	auto ruleIt = _arithmeticRules.find(static_cast<int>(entry.binaryOp));
	if (ruleIt == _arithmeticRules.end()) return;

	std::vector<std::string> inputNames;
	for (const auto &in : entry.inputs) {
		inputNames.push_back(in.name);
	}

	std::string adjSrc = SaveAndZeroAdjoint(dOut, entry.output.glslType);
	ruleIt->second(this, entry, adjSrc, inputNames);
}

void AdjointGenerator::ProcessUnaryOp(const TapeEntry &entry) {
	ProcessBinaryOp(entry); // Same mechanism
}

void AdjointGenerator::ProcessExpressionGradient(const TapeEntry &entry) {
	std::string dOut = _adjTable.Get(entry.output.name);
	if (dOut.empty()) return;

	std::string adjSrc = SaveAndZeroAdjoint(dOut, entry.output.glslType);
	const size_t count = std::min(entry.inputs.size(), entry.inputGradExprs.size());
	for (size_t i = 0; i < count; i++) {
		EmitAccumulate(entry.inputs[i].name,
					   std::format("({})*({})", adjSrc, entry.inputGradExprs[i]));
	}
}

void AdjointGenerator::ProcessIntrinsic1(const TapeEntry &entry) {
	std::string dOut = _adjTable.Get(entry.output.name);
	if (dOut.empty()) return;

	auto ruleIt = _intrinsicRules.find(entry.intrinsicName);
	if (ruleIt == _intrinsicRules.end()) return;

	std::vector<std::string> inputNames;
	for (const auto &in : entry.inputs) inputNames.push_back(in.name);

	std::string adjSrc = SaveAndZeroAdjoint(dOut, entry.output.glslType);
	ruleIt->second(this, entry, adjSrc, inputNames);
}

void AdjointGenerator::ProcessIntrinsic2(const TapeEntry &entry) {
	ProcessIntrinsic1(entry); // Same dispatch through intrinsic name
}

void AdjointGenerator::ProcessIntrinsic3(const TapeEntry &entry) {
	ProcessIntrinsic1(entry); // Same dispatch through intrinsic name
}

void AdjointGenerator::ProcessTernary(const TapeEntry &entry) {
	std::string dOut = _adjTable.Get(entry.output.name);
	if (dOut.empty()) return;
	if (entry.inputs.size() < 3) return;

	const auto &cond = entry.inputs[0];
	const auto &trueVal = entry.inputs[1];
	const auto &falseVal = entry.inputs[2];

	std::string adjTrue = _adjTable.Get(trueVal.name);
	std::string adjFalse = _adjTable.Get(falseVal.name);

	std::string adjSrc = SaveAndZeroAdjoint(dOut, entry.output.glslType);

	if (!adjTrue.empty()) {
		EmitLine(
			std::format("{} += ({})?({}):{}(0);", adjTrue, cond.name, adjSrc, ZeroOf(trueVal.glslType)));
	}
	if (!adjFalse.empty()) {
		EmitLine(
			std::format("{} += ({})?{}(0):({});", adjFalse, cond.name, ZeroOf(falseVal.glslType), adjSrc));
	}
}

void AdjointGenerator::ProcessCompoundAssign(const TapeEntry &entry) {
	std::string dLhs = _adjTable.Get(entry.output.name);
	if (dLhs.empty()) return;
	if (entry.inputs.size() < 2) return;

	const auto &rhs = entry.inputs[1]; // inputs[0] is the LHS itself
	std::string adjRhs = _adjTable.Get(rhs.name);
	if (adjRhs.empty()) return;

	switch (entry.compoundOp) {
	case GPU::IR::Node::CompoundAssignmentCode::AddAssign:
		EmitLine(std::format("{} += {};", adjRhs, dLhs));
		break;
	case GPU::IR::Node::CompoundAssignmentCode::SubAssign:
		EmitLine(std::format("{} += -({});", adjRhs, dLhs));
		break;
	case GPU::IR::Node::CompoundAssignmentCode::MulAssign: {
		// Backprop through a *= b:
		//   d_b += d_a_new * a_old = d_a * a / b
		//   adj_a *= b  (transform d_a_new -> d_a_old)
		EmitLine(std::format("{} += {} * {} / {};", adjRhs, dLhs,
						 entry.output.name, rhs.name));
		EmitLine(std::format("{} *= {};", dLhs, rhs.name));
		break;
	}
	case GPU::IR::Node::CompoundAssignmentCode::DivAssign: {
		// Backprop through a /= b:
		//   d_b += d_a_new * (-a_old / b^2) = -d_a * a / b
		//   adj_a /= b  (transform d_a_new -> d_a_old)
		EmitLine(std::format("{} += -({} * {} / {});", adjRhs, dLhs,
						 entry.output.name, rhs.name));
		EmitLine(std::format("{} /= {};", dLhs, rhs.name));
		break;
	}
	default:
		break;
	}
}

// =============================================================================
// Helpers
// =============================================================================

void AdjointGenerator::EmitAccumulate(const std::string &inputName, const std::string &gradExpr) {
	std::string adjName = _adjTable.Get(inputName);
	if (adjName.empty()) return;
	EmitLine(std::format("{} += {};", ResolveAliases(adjName), ResolveAliases(gradExpr)));
}

void AdjointGenerator::EmitLine(const std::string &line) {
	if (!_controlStack.empty()) {
		_controlStack.back().lines.push_back(line);
	} else {
		_bodyLines.push_back(line);
	}
}

std::vector<std::string> &AdjointGenerator::ActiveLines() {
	if (!_controlStack.empty()) {
		return _controlStack.back().lines;
	}
	return _bodyLines;
}

void AdjointGenerator::PushControlFrame() {
	_controlStack.push_back({});
}

void AdjointGenerator::PopControlFrameAndWrap(const TapeEntry &beginEntry) {
	if (_controlStack.empty()) return;

	auto frame = std::move(_controlStack.back());
	_controlStack.pop_back();

	switch (beginEntry.controlFlowKind) {
	case ControlFlowKind::IfBranch: {
		// if (cond) { frame.lines }
		std::string block = std::format("if ({}) {{\n", beginEntry.conditionVarName);
		for (const auto &line : frame.lines) {
			block += "    " + line + "\n";
		}
		block += "}";
		// Prepend if-block to chain blocks
		frame.chainBlocks.insert(frame.chainBlocks.begin(), block);
		// Emit all chain blocks
		for (const auto &cb : frame.chainBlocks) {
			EmitLine(cb);
		}
		break;
	}
	case ControlFlowKind::ElifBranch: {
		// else if (cond) { frame.lines }
		std::string block = std::format(" else if ({}) {{\n", beginEntry.conditionVarName);
		for (const auto &line : frame.lines) {
			block += "    " + line + "\n";
		}
		block += "}";
		frame.chainBlocks.insert(frame.chainBlocks.begin(), block);
		// Push frame back with updated chain blocks (still collecting) frame.lines.clear();
		_controlStack.push_back(std::move(frame));
		break;
	}
	case ControlFlowKind::ElseBranch: {
		// else { frame.lines }
		std::string block = " else {\n";
		for (const auto &line : frame.lines) {
			block += "    " + line + "\n";
		}
		block += "}";
		frame.chainBlocks.insert(frame.chainBlocks.begin(), block); frame.lines.clear();
		_controlStack.push_back(std::move(frame));
		break;
	}
	case ControlFlowKind::ForLoop: {
		// Reversed for loop: for (int var = end-1; var >= start; var -= step)
		std::string forHeader;
		if (!beginEntry.forVarName.empty()) {
			forHeader = std::format("for (int {} = ({}) - 1; {} >= ({}); {} -= ({})) {{\n",
				beginEntry.forVarName, beginEntry.forEnd,
				beginEntry.forVarName, beginEntry.forStart,
				beginEntry.forVarName, beginEntry.forStep);
		} else {
			forHeader = "for (int _i = 0; _i < 1; _i++) {\n";
		}
		EmitLine(forHeader);
		for (const auto &line : frame.lines) {
			EmitLine("    " + line);
		}
		EmitLine("}");
		break;
	}
	}
}

void AdjointGenerator::ProcessControlFlowBegin(const TapeEntry &entry) {
	PopControlFrameAndWrap(entry);
}

void AdjointGenerator::ProcessControlFlowEnd() {
	PushControlFrame();
}

std::string AdjointGenerator::Adj(const std::string &varName) {
	return _adjTable.Get(varName);
}

std::string AdjointGenerator::ZeroOf(const std::string &glslType) {
	if (glslType == "float") return "0.0";
	if (glslType == "int") return "0";
	if (glslType == "bool") return "false";
	if (glslType == "vec2") return "vec2(0.0)";
	if (glslType == "vec3") return "vec3(0.0)";
	if (glslType == "vec4") return "vec4(0.0)";
	if (glslType == "ivec2") return "ivec2(0)";
	if (glslType == "ivec3") return "ivec3(0)";
	if (glslType == "ivec4") return "ivec4(0)";
	if (glslType == "mat2") return "mat2(0.0)";
	if (glslType == "mat3") return "mat3(0.0)";
	if (glslType == "mat4") return "mat4(0.0)";
	if (glslType == "mat2x3") return "mat2x3(0.0)";
	if (glslType == "mat2x4") return "mat2x4(0.0)";
	if (glslType == "mat3x2") return "mat3x2(0.0)";
	if (glslType == "mat3x4") return "mat3x4(0.0)";
	if (glslType == "mat4x2") return "mat4x2(0.0)";
	if (glslType == "mat4x3") return "mat4x3(0.0)";
	return "0.0";
}


// =============================================================================
// Save-and-zero helper
// =============================================================================

std::string AdjointGenerator::SaveAndZeroAdjoint(const std::string &adjName, const std::string &glslType) {
	std::string tmpName = std::format("_adj{}_", _tmpCounter++);
	EmitLine(std::format("float {} = {};", tmpName, adjName));
	EmitLine(std::format("{} = {};", adjName, ZeroOf(glslType)));
	return tmpName;
}

// =============================================================================
// Alias resolution
// =============================================================================

void AdjointGenerator::BuildAliasMap() {
	if (!_tape) return;

	_aliasMap.clear();

	// Helper: check if a name is a pure scalar variable (e.g. "v95")
	auto isScalarVar = [](const std::string &name) -> bool {
		if (name.empty() || name[0] != 'v') return false;
		for (size_t i = 1; i < name.size(); i++) {
			if (name[i] < '0' || name[i] > '9') return false;
		}
		return true;
	};

	// Build direct alias map from simple copy operations (v95 = v93 recorded as Add-with-zero)
	std::unordered_map<std::string, std::string> directAliases;
	for (const auto &entry : _tape->Entries()) {
		if (entry.kind == TapeOpKind::BinaryOp &&
			entry.binaryOp == GPU::IR::Node::OperationCode::Add &&
			entry.inputs.size() == 2 &&
			entry.inputs[1].name == "0" &&
			isScalarVar(entry.output.name) &&
			isScalarVar(entry.inputs[0].name)) {
			directAliases[entry.output.name] = entry.inputs[0].name;
		}
	}

	if (directAliases.empty()) return;

	// Compute transitive closure: v95 -> v93 -> v91 -> v85  =>  v95 -> v85
	for (const auto &[alias, target] : directAliases) {
		std::string canonical = target;
		std::unordered_set<std::string> visited{alias};
		while (directAliases.count(canonical) && visited.insert(canonical).second) {
			canonical = directAliases[canonical];
		}
		if (canonical != alias) {
			_aliasMap[alias] = canonical;
		}
	}
}

std::string AdjointGenerator::ResolveAliases(const std::string &expr) const {
	if (_aliasMap.empty()) return expr;

	// Sort aliases by name length (longest first) to avoid partial replacements
	std::vector<std::pair<std::string, std::string>> sortedAliases(
		_aliasMap.begin(), _aliasMap.end());
	std::sort(sortedAliases.begin(), sortedAliases.end(),
		[](const auto &a, const auto &b) {
			return a.first.size() > b.first.size();
		});

	std::string result = expr;
	for (const auto &[alias, canonical] : sortedAliases) {
		size_t pos = 0;
		while ((pos = result.find(alias, pos)) != std::string::npos) {
			// Word-boundary check: character before must not be [a-zA-Z0-9_]
			bool leftOk = (pos == 0) ||
				!(std::isalnum(static_cast<unsigned char>(result[pos - 1])) || result[pos - 1] == '_');
			// Character after must not be [a-zA-Z0-9_]
			size_t endPos = pos + alias.size();
			bool rightOk = (endPos >= result.size()) ||
				!(std::isalnum(static_cast<unsigned char>(result[endPos])) || result[endPos] == '_');

			if (leftOk && rightOk) {
				result.replace(pos, alias.size(), canonical);
				pos += canonical.size();
			} else {
				pos += alias.size();
			}
		}
	}
	return result;
}

// =============================================================================
// Callable and Return processing
// =============================================================================

void AdjointGenerator::ProcessCall(const TapeEntry &entry) {
	// Get the adjoint of the call output from the main adjoint table
	std::string dOut = _adjTable.Get(entry.output.name);
	if (dOut.empty()) return;
	if (!_tape) return;

	// O(1) sub-tape lookup via pre-built index map
	auto mapIt = _callIndexMap.find(entry.id);
	if (mapIt == _callIndexMap.end()) return;
	int callIndex = mapIt->second;
	if (callIndex >= static_cast<int>(_tape->SubTapeCount())) return;

	const auto &subTape = _tape->SubTape(callIndex);
	if (subTape.Size() == 0) return;

	// Build a name-remapped copy of the sub-tape for the adjoint generator.
	// Parameter names (p0, p1, ...) are mapped to the caller's input names.
	// Internal variable names are prefixed for uniqueness.
	std::string prefix = "_ca" + std::to_string(callIndex) + "_";
	std::unordered_map<std::string, std::string> nameMap;
	// Map parameters: p0 -> inputs[0].name, p1 -> inputs[1].name, ...
	for (size_t pi = 0; pi < entry.inputs.size(); pi++) {
		nameMap["p" + std::to_string(pi)] = entry.inputs[pi].name;
	}

	// Find the return variable from the sub-tape
	std::string retVarName;
	for (size_t i = 0; i < subTape.Size(); i++) {
		if (subTape[i].kind == TapeOpKind::Return && !subTape[i].output.name.empty()) {
			retVarName = subTape[i].output.name;
			break;
		}
	}

	// Create a remapped tape with renamed variables
	GradientTape remappedTape;
	for (size_t i = 0; i < subTape.Size(); i++) {
		TapeEntry se = subTape[i];

		// Remap output name
		std::string origOutName = se.output.name;
		if (!origOutName.empty()) {
			auto nit = nameMap.find(origOutName);
			if (nit != nameMap.end()) {
				se.output.name = nit->second;
			} else if (origOutName != retVarName || retVarName.empty()) {
				se.output.name = prefix + origOutName;
			}
			// For the return variable, we handle it specially below
		}

		// Remap input names
		for (auto &in : se.inputs) {
			if (!in.name.empty()) {
				auto nit = nameMap.find(in.name);
				if (nit != nameMap.end()) {
					in.name = nit->second;
				} else {
					in.name = prefix + in.name;
				}
			}
		}

		// Skip Return entries — we handle them specially
		if (se.kind == TapeOpKind::Return) continue;

		// For binary ops with literal operands, keep them as-is
		// Check 'forVarName' etc. in control flow entries
		if (se.kind == TapeOpKind::ControlFlowBegin) {
			if (!se.conditionVarName.empty()) {
				auto nit = nameMap.find(se.conditionVarName);
				if (nit != nameMap.end()) se.conditionVarName = nit->second;
				else se.conditionVarName = prefix + se.conditionVarName;
			}
		}

		remappedTape.RecordRemapped(se);
	}

	// Deep-copy sub-tapes so nested callable bodies work
	remappedTape.CloneSubTapesFrom(subTape);

	// Mark the return variable as the "loss" for the remapped tape
	if (!retVarName.empty()) {
		std::string mappedRetName = prefix + retVarName;
		// Also check if retVarName is a parameter
		auto nit = nameMap.find(retVarName);
		if (nit != nameMap.end()) mappedRetName = nit->second;
		remappedTape.MarkLoss(mappedRetName, "float");
		// Seed: the adjoint of the return variable = dOut
		// We'll handle this manually after generation
	}

	// Register parameters in the remapped tape
	for (size_t pi = 0; pi < entry.inputs.size(); pi++) {
		std::string pName = entry.inputs[pi].name;
		std::string pType = entry.inputs[pi].glslType;
		remappedTape.RegisterParameter(pName, pType);
	}

	// Generate adjoint body for the remapped sub-tape
	AdjointGenerator subGen;
	AdjointBody subBody = subGen.GenerateBody(remappedTape, false);

	// Emit declarations for local adjoint variables (prefixed ones)
	for (const auto &[adjName, glslType] : subBody.declarations) {
		// Filter: only emit local ones (not parameter adjoints which are already
		// in the main adjoint table)
		if (adjName.find(prefix) == 0) {
			// Add to main adjTable so EmitLine works
			_adjTable.GetOrCreate(adjName, glslType);
		} else {
			// This is a parameter adjoint — already in the main adjTable
		}
	}

	// Emit the sub-body lines, replacing the d_ret seed with dOut
	for (const auto &line : subBody.lines) {
		// Find the seed line (adj_of_loss = float(1.0)) and replace with dOut
		std::string seedName = prefix + retVarName;
		auto nit = nameMap.find(retVarName);
		if (nit != nameMap.end()) seedName = nit->second;
		
		std::string adjRetName = subGen.GetAdjointTable().Get(
			nit != nameMap.end() ? nit->second : (prefix + retVarName));
		
		std::string expectedSeed = adjRetName + " = float(1.0);";
		if (line == expectedSeed) {
			EmitLine(adjRetName + " += " + dOut + ";");
		} else {
			EmitLine(line);
		}
	}
}

void AdjointGenerator::ProcessReturn(const TapeEntry &entry) {
	// Return entries only appear in sub-tapes (callable bodies).
	// They mark which variable was returned.
	// In the inline approach, this is handled by mapping the returned
	// variable to the d_ret adjoint from the caller.
	// For now, this is a no-op in the main tape pass.
}
// =============================================================================
// Final Assembly
// =============================================================================

std::string AdjointGenerator::Assemble() {
	std::ostringstream code;

	// Compute shader header
	code << "#version 430 core\n";
	code << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

	// Main function
	code << "void main() {\n";

	// Adjoint variable declarations with zero initialization
	for (const auto &[adjName, glslType] : _adjTable.AllDeclarations()) {
		code << std::format("    {} {} = {}(0);\n", glslType, adjName, glslType);
	}

	code << "\n";

	// Body: gradient accumulation statements
	for (const auto &line : _bodyLines) {
		code << "    " << line << "\n";
	}

	// Parameter gradient write-back
	for (const auto &[paramName, adjName] : _paramWritebacks) {
		code << std::format("    {} = {};\n", paramName, adjName);
	}

	code << "}\n";

	return code.str();
}

} // namespace GPU::AD
