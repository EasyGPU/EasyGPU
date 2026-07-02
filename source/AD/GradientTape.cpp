/**
 * @file GradientTape.cpp
 * @brief Implementation of the gradient tape recording engine.
 *
 * Hooks into Builder::Build() to record every differentiable operation
 * during the forward pass. Analyzes IR node types to extract operation
 * metadata (op kind, input/output variable names, types).
 */

#include <AD/GradientTape.h>
#include <IR/Builder/Builder.h>
#include <IR/Node/Call.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/CompoundAssignment.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/Load.h>
#include <IR/Node/LoadLocalArray.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/MemberAccess.h>
#include <IR/Node/Node.h>
#include <IR/Node/Operation.h>
#include <IR/Node/Return.h>
#include <IR/Node/Store.h>
#include <IR/Node/Ternary.h>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <functional>
#include <sstream>

namespace GPU::AD {

// =============================================================================
// Helper: construct a TapeEntry by setting fields explicitly
// =============================================================================

namespace {

TapeEntry MakeEntry(int32_t id, TapeOpKind kind, const TapeVar &output, const std::vector<TapeVar> &inputs) {
	TapeEntry e;
	e.id	 = id;
	e.kind	 = kind;
	e.output = output;
	e.inputs = inputs;
	return e;
}

bool IsNumericLiteralName(const std::string &name) {
	if (name.empty())
		return false;
	char *end = nullptr;
	std::strtof(name.c_str(), &end);
	if (end == name.c_str())
		return false;
	if (*end == '\0')
		return true;
	return (end[0] == 'f' || end[0] == 'F' || end[0] == 'u' || end[0] == 'U') && end[1] == '\0';
}

bool IsConstantConstructorExpression(const std::string &name, size_t typeLength) {
	if (name.size() <= typeLength + 2 || name[typeLength] != '(' || name.back() != ')') {
		return false;
	}
	for (size_t i = typeLength + 1; i + 1 < name.size(); i++) {
		const unsigned char c = static_cast<unsigned char>(name[i]);
		if (std::isalpha(c) || c == '_' || c == '[' || c == ']') {
			return false;
		}
	}
	return true;
}

bool IsLiteralName(const std::string &name) {
	if (name.empty())
		return true;
	if (name == "true" || name == "false")
		return true;
	if (IsNumericLiteralName(name))
		return true;
	static const char *glslTypes[] = {"float", "int",	"uint",	 "bool",  "vec2",  "vec3",	"vec4",	 "ivec2",
									  "ivec3", "ivec4", "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
									  "mat2",  "mat3",	"mat4",	 "dvec2", "dvec3", "dvec4", "dmat2", "dmat3",
									  "dmat4"};
	for (const char *t : glslTypes) {
		size_t len = std::char_traits<char>::length(t);
		if (name.compare(0, len, t) == 0 && IsConstantConstructorExpression(name, len)) {
			return true;
		}
	}
	return false;
}

std::string BuildNodeExpression(const GPU::IR::Node::Node &node) {
	return GPU::IR::Builder::Builder::Get().BuildNode(node);
}

int VectorSize(const std::string &type) {
	if (type == "vec2" || type == "ivec2")
		return 2;
	if (type == "vec3" || type == "ivec3")
		return 3;
	if (type == "vec4" || type == "ivec4")
		return 4;
	return 0;
}

bool IsDifferentiableType(const std::string &type) {
	return type == "float" || type == "vec2" || type == "vec3" || type == "vec4";
}

int SwizzleComponentIndex(char component) {
	switch (component) {
	case 'x':
	case 'r':
		return 0;
	case 'y':
	case 'g':
		return 1;
	case 'z':
	case 'b':
		return 2;
	case 'w':
	case 'a':
		return 3;
	default:
		return -1;
	}
}

char VectorComponentName(size_t index) {
	static constexpr char components[] = {'x', 'y', 'z', 'w'};
	return index < (sizeof(components) / sizeof(components[0])) ? components[index] : 'x';
}

bool IsVectorConstructorName(const std::string &name) {
	return name == "vec2" || name == "vec3" || name == "vec4";
}

std::string SwizzleComponentExpr(const std::string &expr, size_t componentIndex) {
	return std::format("({}).{}", expr, VectorComponentName(componentIndex));
}

std::string SumSwizzleComponents(const std::string &expr, int componentCount) {
	std::string sum;
	for (int i = 0; i < componentCount; i++) {
		auto component = SwizzleComponentExpr(expr, static_cast<size_t>(i));
		if (sum.empty()) {
			sum = std::move(component);
		} else {
			sum = std::format("({})+({})", sum, component);
		}
	}
	return sum.empty() ? "0.0" : sum;
}

std::string SingleArgumentIntrinsicDerivative(const std::string &intrinsicName, const std::string &argExpr) {
	if (intrinsicName == "sin")
		return std::format("cos({})", argExpr);
	if (intrinsicName == "cos")
		return std::format("(-sin({}))", argExpr);
	if (intrinsicName == "exp")
		return std::format("exp({})", argExpr);
	if (intrinsicName == "log")
		return std::format("1.0/({})", argExpr);
	if (intrinsicName == "sqrt")
		return std::format("1.0/(2.0*sqrt({}))", argExpr);
	if (intrinsicName == "abs")
		return std::format("sign({})", argExpr);
	if (intrinsicName == "tan")
		return std::format("(1.0+tan({})*tan({}))", argExpr, argExpr);
	if (intrinsicName == "asin")
		return std::format("1.0/sqrt(1.0-({})*({}))", argExpr, argExpr);
	if (intrinsicName == "acos")
		return std::format("(-1.0/sqrt(1.0-({})*({})))", argExpr, argExpr);
	if (intrinsicName == "atan")
		return std::format("1.0/(1.0+({})*({}))", argExpr, argExpr);
	if (intrinsicName == "sinh")
		return std::format("cosh({})", argExpr);
	if (intrinsicName == "cosh")
		return std::format("sinh({})", argExpr);
	if (intrinsicName == "tanh")
		return std::format("(1.0-tanh({})*tanh({}))", argExpr, argExpr);
	if (intrinsicName == "exp2")
		return std::format("log(2.0)*exp2({})", argExpr);
	if (intrinsicName == "log2")
		return std::format("1.0/(({})*log(2.0))", argExpr);
	if (intrinsicName == "inversesqrt")
		return std::format("-0.5/(({})*sqrt({}))", argExpr, argExpr);
	if (intrinsicName == "fract")
		return "1.0";
	if (intrinsicName == "radians")
		return "0.01745329252";
	if (intrinsicName == "degrees")
		return "57.295779513";
	if (intrinsicName == "asinh")
		return std::format("1.0/sqrt(({})*({})+1.0)", argExpr, argExpr);
	if (intrinsicName == "acosh")
		return std::format("1.0/sqrt(({})*({})-1.0)", argExpr, argExpr);
	if (intrinsicName == "atanh")
		return std::format("1.0/(1.0-({})*({}))", argExpr, argExpr);
	return "";
}

} // anonymous namespace

// =============================================================================
// Public interface
// =============================================================================

void GradientTape::Record(const GPU::IR::Node::Node &node, bool isStatement) {
	// If we're inside a callable body, redirect to the active sub-tape
	if (GPU::IR::Builder::Builder::Get().IsInCallableBody()) {
		if (_currentSubTape) {
			_currentSubTape->RecordDirect(node, isStatement);
		}
		return;
	}
	RecordDirect(node, isStatement);
}

void GradientTape::RecordDirect(const GPU::IR::Node::Node &node, bool isStatement) {
	switch (node.Type()) {
	case GPU::IR::Node::NodeType::Store:
		RecordStore(static_cast<const GPU::IR::Node::StoreNode &>(node));
		break;

	case GPU::IR::Node::NodeType::CompoundAssignment:
		RecordCompoundAssignment(static_cast<const GPU::IR::Node::CompoundAssignmentNode &>(node));
		break;

	case GPU::IR::Node::NodeType::LocalVariable:
		RecordLocalVariable(static_cast<const GPU::IR::Node::LocalVariableNode &>(node));
		break;

	case GPU::IR::Node::NodeType::Return:
		RecordReturn(static_cast<const GPU::IR::Node::ReturnNode &>(node));
		break;

	default:
		break;
	}
}

void GradientTape::RecordRemapped(const TapeEntry &entry) {
	// Add a pre-remapped entry with an auto-assigned ID.
	// Used by AdjointGenerator to clone and remap sub-tape entries.
	TapeEntry e = entry;
	e.id		= _nextId++;
	_entries.push_back(std::move(e));
}

void GradientTape::RegisterParameter(const std::string &name, const std::string &glslType) {
	if (_parameters.count(name) == 0) {
		_paramList.emplace_back(name, glslType);
	}
	_parameters[name] = glslType;
	_activeNames.insert(name);
	if (!_varTypes.count(name)) {
		_varTypes[name] = glslType;
	}
}

void GradientTape::RegisterBufferParameter(const std::string &bufferName, const std::string &elementType,
										   size_t elementCount) {
	if (bufferName.empty() || elementCount == 0)
		return;

	auto it = _bufferParameters.find(bufferName);
	if (it == _bufferParameters.end()) {
		BufferParam param;
		param.bufferName	= bufferName;
		param.elementType	= elementType;
		param.elementCount	= elementCount;
		_bufferParamList.push_back(param);
		_bufferParameters.emplace(bufferName, std::move(param));
	} else {
		it->second.elementType	= elementType;
		it->second.elementCount = std::max(it->second.elementCount, elementCount);
		for (auto &param : _bufferParamList) {
			if (param.bufferName == bufferName) {
				param.elementType	= elementType;
				param.elementCount = it->second.elementCount;
				break;
			}
		}
	}

	_activeNames.insert(bufferName);
	if (!_varTypes.count(bufferName)) {
		_varTypes[bufferName] = elementType;
	}
}

void GradientTape::RegisterBufferAdjointStorage(const std::string &bufferName, const std::string &elementType,
												size_t elementCount) {
	if (bufferName.empty() || elementCount == 0)
		return;

	auto it = _bufferAdjointStorages.find(bufferName);
	if (it == _bufferAdjointStorages.end()) {
		BufferAdjointStorage storage;
		storage.bufferName	 = bufferName;
		storage.elementType	 = elementType;
		storage.elementCount = elementCount;
		_bufferAdjointStorageList.push_back(storage);
		_bufferAdjointStorages.emplace(bufferName, std::move(storage));
	} else {
		it->second.elementType	= elementType;
		it->second.elementCount = std::max(it->second.elementCount, elementCount);
		for (auto &storage : _bufferAdjointStorageList) {
			if (storage.bufferName == bufferName) {
				storage.elementType	 = elementType;
				storage.elementCount = it->second.elementCount;
				break;
			}
		}
	}

	if (!_varTypes.count(bufferName)) {
		_varTypes[bufferName] = elementType;
	}
}

bool GradientTape::IsParameter(const std::string &name) const {
	if (_parameters.count(name) > 0)
		return true;
	auto bpos = name.find('[');
	if (bpos != std::string::npos) {
		return _bufferParameters.count(name.substr(0, bpos)) > 0;
	}
	return _bufferParameters.count(name) > 0;
}

void GradientTape::MarkLoss(const std::string &name, const std::string &glslType) {
	_lossVar = TapeVar{name, glslType, false};
	_activeNames.insert(name);
	if (!_varTypes.count(name)) {
		_varTypes[name] = glslType;
	}
}

const std::string *GradientTape::GetVarType(const std::string &name) const {
	auto it = _varTypes.find(name);
	return it != _varTypes.end() ? &it->second : nullptr;
}

// =============================================================================
// Internal: Node analysis
// =============================================================================

void GradientTape::RecordStore(const GPU::IR::Node::StoreNode &store) {
	const auto *lhs		= store.LHS();
	const auto *rhs		= store.RHS();

	std::string outName = ExtractVarName(*lhs);
	std::string outType;
	if (auto *tp = GetVarType(outName))
		outType = *tp;
	else
		outType = "float";

	TapeVar output{outName, outType, IsParameter(outName)};

	RecordAssignmentRhs(*rhs, output);
}

void GradientTape::RecordAssignmentRhs(const GPU::IR::Node::Node &rhs, const TapeVar &output) {
	auto recordAlias = [this, &output](const GPU::IR::Node::Node &node) {
		// Record if the RHS is an actual variable load, not a literal/constant.
		std::string inName = TryExtractVarName(node);
		if (inName.empty()) {
			// TryExtractVarName filters names containing '(' (which includes
			// buffer accesses with complex index expressions like
			// buf_W[int(tokenId)*E+int(d)]). Use ExtractVarName to capture
			// the full buffer access string so gradients can flow back
			// through buffer reads.
			inName = ExtractVarName(node);
		}
		if (!inName.empty() && inName != output.name) {
			std::string inType;
			if (auto *t = GetVarType(inName))
				inType = *t;
			else
				inType = output.glslType.empty() ? "float" : output.glslType;

			auto entry	   = MakeEntry(_nextId++, TapeOpKind::BinaryOp, output,
									   {TapeVar{inName, inType, IsParameter(inName)}, TapeVar{"0", "float", false}});
			entry.binaryOp = GPU::IR::Node::OperationCode::Add;
			entry.forwardExpr = BuildNodeExpression(node);
			if (entry.forwardExpr.empty()) {
				entry.forwardExpr = inName;
			}
			_entries.push_back(std::move(entry));

			PropagateActive(output, {TapeVar{inName, inType, IsParameter(inName)}});
		}
		// If inName is still empty, it's a literal/uniform value
		// (e.g., `a = 2.0f;` where RHS is LoadUniform("float(2.0)")).
		// These have zero gradient and don't need tape entries.
	};

	switch (rhs.Type()) {
	case GPU::IR::Node::NodeType::Operation:
		RecordOperation(static_cast<const GPU::IR::Node::OperationNode &>(rhs), output);
		break;

	case GPU::IR::Node::NodeType::CallInst:
		RecordIntrinsic(static_cast<const GPU::IR::Node::IntrinsicCallNode &>(rhs), output);
		break;

	case GPU::IR::Node::NodeType::Ternary:
		RecordTernary(static_cast<const GPU::IR::Node::TernaryNode &>(rhs), output);
		break;

	case GPU::IR::Node::NodeType::Load: {
		recordAlias(rhs);
		break;
	}

	case GPU::IR::Node::NodeType::Call: {
		RecordCall(static_cast<const GPU::IR::Node::CallNode &>(rhs), output);
		break;
	}

	case GPU::IR::Node::NodeType::MemberAccess:
		RecordMemberAccess(static_cast<const GPU::IR::Node::MemberAccessNode &>(rhs), output);
		break;

	case GPU::IR::Node::NodeType::ArrayAccess:
		recordAlias(rhs);
		break;

	default:
		break;
	}
}

void GradientTape::RecordMemberAccess(const GPU::IR::Node::MemberAccessNode &node, const TapeVar &output) {
	if (node.LHS() == nullptr || node.RHS() == nullptr)
		return;

	std::string baseName = ExtractVarName(*node.LHS());
	if (baseName.empty())
		return;

	std::string memberName = ExtractMemberName(*node.RHS());
	if (memberName.empty())
		return;

	std::string baseType;
	if (auto *t = GetVarType(baseName))
		baseType = *t;
	else
		baseType = InferNodeType(*node.LHS());

	std::vector<TapeVar> inputs{TapeVar{baseName, baseType, IsParameter(baseName)}};
	auto				 entry = MakeEntry(_nextId++, TapeOpKind::ExpressionGradient, output, inputs);
	entry.inputGradExprs.push_back(BuildSwizzleScatterExpression(memberName, "1.0", baseType));
	entry.inputGradTypes.push_back(baseType);
	entry.forwardExpr = BuildNodeExpression(node);
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

void GradientTape::RecordCompoundAssignment(const GPU::IR::Node::CompoundAssignmentNode &node) {
	std::string name	= ExtractVarName(*node.LHS());
	std::string rhsName = TryExtractVarName(*node.RHS());
	std::string type;
	if (auto *t = GetVarType(name))
		type = *t;
	else
		type = "float";

	auto code = node.Code();
	if (code != GPU::IR::Node::CompoundAssignmentCode::AddAssign &&
		code != GPU::IR::Node::CompoundAssignmentCode::SubAssign &&
		code != GPU::IR::Node::CompoundAssignmentCode::MulAssign &&
		code != GPU::IR::Node::CompoundAssignmentCode::DivAssign) {
		return;
	}

	std::string rhsType;
	if (auto *t = GetVarType(rhsName))
		rhsType = *t;
	else
		rhsType = "float";

	TapeVar output{name, type, IsParameter(name)};
	TapeVar rhsVar{rhsName, rhsType, IsParameter(rhsName)};

	auto	entry	 = MakeEntry(_nextId++, TapeOpKind::CompoundAssign, output, {output, rhsVar});
	entry.compoundOp = code;
	entry.forwardExpr = BuildNodeExpression(*node.RHS());
	_entries.push_back(std::move(entry));

	if (IsActive(name)) {
		_activeNames.insert(rhsName);
	}
}

void GradientTape::RecordOperation(const GPU::IR::Node::OperationNode &op, const TapeVar &output) {
	auto code = op.Code();

	switch (code) {
	case GPU::IR::Node::OperationCode::Add:
	case GPU::IR::Node::OperationCode::Sub:
	case GPU::IR::Node::OperationCode::Mul:
	case GPU::IR::Node::OperationCode::Div:
	case GPU::IR::Node::OperationCode::Neg:
		break;
	default:
		return;
	}

	std::vector<TapeVar>	 inputs;
	std::vector<std::string> inputGradExprs;
	std::vector<std::string> inputGradTypes;

	CollectExpressionLeaves(op, "1.0", output.glslType, inputs, inputGradExprs, inputGradTypes);
	if (inputs.empty())
		return;

	auto entry			 = MakeEntry(_nextId++, TapeOpKind::ExpressionGradient, output, inputs);
	entry.binaryOp		 = code;
	entry.inputGradExprs = std::move(inputGradExprs);
	entry.inputGradTypes = std::move(inputGradTypes);
	entry.forwardExpr	 = BuildNodeExpression(op);
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

void GradientTape::RecordIntrinsic(const GPU::IR::Node::IntrinsicCallNode &node, const TapeVar &output) {
	const auto &params	= node.Parameter();
	size_t		nParams = params.size();
	std::string intrinsicName(node.Name());

	if (nParams == 1 && params[0]) {
		const auto &param = *params[0];
		std::string paramName = TryExtractVarName(param);
		if (paramName.empty())
			paramName = ExtractVarName(param);
		if (paramName.empty()) {
			std::string paramExpr = GPU::IR::Builder::Builder::Get().BuildNode(param);
			if (paramExpr.empty()) {
				return;
			}
			std::string paramType = InferNodeType(param);
			if (!IsDifferentiableType(paramType)) {
				return;
			}

			std::vector<TapeVar>	 inputs;
			std::vector<std::string> inputGradExprs;
			std::vector<std::string> inputGradTypes;
			CollectExpressionLeaves(param, "1.0", paramType, inputs, inputGradExprs, inputGradTypes);
			if (!inputs.empty()) {
				const std::string paramTempName = std::format("_ad_expr{}", _nextId);
				TapeVar			  paramOutput{paramTempName, paramType, false};
				auto exprEntry = MakeEntry(_nextId++, TapeOpKind::ExpressionGradient, paramOutput, inputs);
				exprEntry.inputGradExprs = std::move(inputGradExprs);
				exprEntry.inputGradTypes = std::move(inputGradTypes);
				exprEntry.forwardExpr = paramExpr;
				_entries.push_back(std::move(exprEntry));
				PropagateActive(paramOutput, inputs);

				auto intrinsicEntry = MakeEntry(
					_nextId++,
					nParams == 1 ? TapeOpKind::Intrinsic1 : (nParams == 2 ? TapeOpKind::Intrinsic2 : TapeOpKind::Intrinsic3),
					output,
					{paramOutput});
				intrinsicEntry.intrinsicName = intrinsicName;
				intrinsicEntry.forwardExpr = BuildNodeExpression(node);
				_entries.push_back(std::move(intrinsicEntry));
				PropagateActive(output, {paramOutput});
				return;
			}
		}
	}

	if (nParams == 2 && params[0] && params[1] && (intrinsicName == "min" || intrinsicName == "max")) {
		const auto &a = *params[0];
		const auto &b = *params[1];
		const auto aExpr = GPU::IR::Builder::Builder::Get().BuildNode(a);
		const auto bExpr = GPU::IR::Builder::Builder::Get().BuildNode(b);
		const auto choose = std::format("step({},{})", aExpr, bExpr);
		const auto gradA = intrinsicName == "max" ? std::format("1.0-({})", choose) : choose;
		const auto gradB = intrinsicName == "max" ? choose : std::format("1.0-({})", choose);

		std::vector<TapeVar> inputs;
		std::vector<std::string> inputGradExprs;
		std::vector<std::string> inputGradTypes;
		CollectExpressionLeaves(a, gradA, output.glslType, inputs, inputGradExprs, inputGradTypes);
		CollectExpressionLeaves(b, gradB, output.glslType, inputs, inputGradExprs, inputGradTypes);
		if (!inputs.empty()) {
			auto entry			 = MakeEntry(_nextId++, TapeOpKind::ExpressionGradient, output, inputs);
			entry.intrinsicName = intrinsicName;
			entry.inputGradExprs = std::move(inputGradExprs);
			entry.inputGradTypes = std::move(inputGradTypes);
			entry.forwardExpr	 = BuildNodeExpression(node);
			_entries.push_back(std::move(entry));

			PropagateActive(output, inputs);
			return;
		}
	}

	TapeOpKind	kind;
	switch (nParams) {
	case 1:
		kind = TapeOpKind::Intrinsic1;
		break;
	case 2:
		kind = TapeOpKind::Intrinsic2;
		break;
	default:
		kind = TapeOpKind::Intrinsic3;
		break;
	}

	std::vector<TapeVar> inputs;
	for (const auto &p : params) {
		std::string n = ExtractVarName(*p);
		std::string t;
		if (auto *tp = GetVarType(n))
			t = *tp;
		else
			t = "float";
		inputs.push_back(TapeVar{n, t, IsParameter(n)});
	}

	auto entry			= MakeEntry(_nextId++, kind, output, inputs);
	entry.intrinsicName = intrinsicName;
	entry.forwardExpr	= BuildNodeExpression(node);
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

void GradientTape::RecordTernary(const GPU::IR::Node::TernaryNode &node, const TapeVar &output) {
	auto makeInput = [this](const GPU::IR::Node::Node &operand) -> TapeVar {
		std::string n = ExtractVarName(operand);
		std::string t;
		if (auto *tp = GetVarType(n))
			t = *tp;
		else
			t = "float";
		return TapeVar{n, t, IsParameter(n)};
	};

	std::vector<TapeVar> inputs;
	inputs.push_back(makeInput(*node.Condition()));
	inputs.push_back(makeInput(*node.TrueExpr()));
	inputs.push_back(makeInput(*node.FalseExpr()));

	auto entry = MakeEntry(_nextId++, TapeOpKind::Ternary, output, inputs);
	entry.forwardExpr = BuildNodeExpression(node);
	_entries.push_back(std::move(entry));

	if (IsActive(output.name) || IsParameter(output.name)) {
		_activeNames.insert(output.name);
		_activeNames.insert(inputs[1].name);
		_activeNames.insert(inputs[2].name);
	}
}

void GradientTape::RecordLocalVariable(const GPU::IR::Node::LocalVariableNode &node) {
	std::string name = node.VarName();
	if (node.IsExternal())
		return;
	if (name.find("gl_GlobalInvocationID") != std::string::npos)
		return;

	_varTypes[name] = node.VarType();
	if (node.HasInitializer()) {
		TapeVar output{name, node.VarType(), IsParameter(name)};
		RecordAssignmentRhs(*node.Initializer(), output);
	}
}

void GradientTape::AddExpressionLeaf(const GPU::IR::Node::Node &node, const std::string &coeff,
									 const std::string &coeffType, std::vector<TapeVar> &inputs,
									 std::vector<std::string> &inputGradExprs,
									 std::vector<std::string> &inputGradTypes) {
	if (node.Type() == GPU::IR::Node::NodeType::Call) {
		const auto &call = static_cast<const GPU::IR::Node::CallNode &>(node);
		std::string callExpr = GPU::IR::Builder::Builder::Get().BuildNode(call);
		if (callExpr.empty() || IsLiteralName(callExpr)) {
			return;
		}

		std::vector<TapeVar> callInputs;
		callInputs.reserve(call.Arguments().size());
		for (const auto &arg : call.Arguments()) {
			if (!arg) {
				continue;
			}

			std::string argName = TryExtractVarName(*arg);
			if (argName.empty()) {
				argName = ExtractVarName(*arg);
			}
			if (argName.empty()) {
				argName = GPU::IR::Builder::Builder::Get().BuildNode(*arg);
			}

			std::string argType;
			if (auto *tp = GetVarType(argName)) {
				argType = *tp;
			} else {
				argType = InferNodeType(*arg);
			}
			callInputs.push_back(TapeVar{argName, argType, IsParameter(argName)});
		}

		TapeVar callOutput{callExpr, InferNodeType(call), false};
		auto	callEntry = MakeEntry(_nextId++, TapeOpKind::Call, callOutput, callInputs);
		callEntry.callableFuncName = call.FuncName();
		callEntry.forwardExpr = callExpr;
		_entries.push_back(std::move(callEntry));

		inputs.push_back(callOutput);
		inputGradExprs.push_back(coeff);
		inputGradTypes.push_back(coeffType);
		return;
	}

	if (node.Type() == GPU::IR::Node::NodeType::MemberAccess) {
		const auto &member = static_cast<const GPU::IR::Node::MemberAccessNode &>(node);
		if (member.LHS() != nullptr && member.RHS() != nullptr) {
			std::string baseName = ExtractVarName(*member.LHS());
			std::string memberName = ExtractMemberName(*member.RHS());
			if (!baseName.empty() && !memberName.empty()) {
				std::string baseType;
				if (auto *tp = GetVarType(baseName))
					baseType = *tp;
				else
					baseType = InferNodeType(*member.LHS());

				inputs.push_back(TapeVar{baseName, baseType, IsParameter(baseName)});
				inputGradExprs.push_back(BuildSwizzleScatterExpression(memberName, coeff, baseType));
				inputGradTypes.push_back(baseType);
				return;
			}
		}
	}

	std::string n = TryExtractVarName(node);
	if (n.empty())
		n = ExtractVarName(node);
	if (n.empty())
		n = GPU::IR::Builder::Builder::Get().BuildNode(node);
	if (IsLiteralName(n))
		return;
	std::string t;
	if (auto *tp = GetVarType(n))
		t = *tp;
	else
		t = InferNodeType(node);
	if (!IsDifferentiableType(t))
		return;
	inputs.push_back(TapeVar{n, t, IsParameter(n)});
	inputGradExprs.push_back(coeff);
	inputGradTypes.push_back(coeffType);
}

void GradientTape::CollectExpressionLeaves(const GPU::IR::Node::Node &node, const std::string &upstream,
										   const std::string &upstreamType, std::vector<TapeVar> &inputs,
										   std::vector<std::string> &inputGradExprs,
										   std::vector<std::string> &inputGradTypes) {
	auto nodeExpr = [](const GPU::IR::Node::Node &n) { return GPU::IR::Builder::Builder::Get().BuildNode(n); };

	if (node.Type() == GPU::IR::Node::NodeType::CallInst) {
		const auto &intrinsic = static_cast<const GPU::IR::Node::IntrinsicCallNode &>(node);
		const auto &params	  = intrinsic.Parameter();
		const auto intrinsicName = std::string(intrinsic.Name());
		if (IsVectorConstructorName(intrinsicName)) {
			const int resultSize = VectorSize(intrinsicName);
			if (resultSize <= 0) {
				return;
			}

			if (params.size() == 1 && params[0]) {
				const auto argType = InferNodeType(*params[0]);
				if (VectorSize(argType) <= 0) {
					if (upstream == "1.0" && VectorSize(upstreamType) == resultSize) {
						CollectExpressionLeaves(*params[0], std::format("{}(1.0)", upstreamType), upstreamType, inputs,
												inputGradExprs, inputGradTypes);
					} else {
						CollectExpressionLeaves(*params[0], SumSwizzleComponents(upstream, resultSize), "float", inputs,
												inputGradExprs, inputGradTypes);
					}
					return;
				}
			}

			int componentOffset = 0;
			for (const auto &param : params) {
				if (!param || componentOffset >= resultSize) {
					continue;
				}

				const auto argType = InferNodeType(*param);
				const int  argSize = std::max(1, VectorSize(argType));
				const int  take = std::min(argSize, resultSize - componentOffset);
				if (take <= 0) {
					continue;
				}

				if (argSize == 1) {
					if (upstream == "1.0" && VectorSize(upstreamType) == resultSize) {
						const std::string memberName(1, VectorComponentName(static_cast<size_t>(componentOffset)));
						CollectExpressionLeaves(
							*param,
							BuildSwizzleScatterExpression(memberName, "1.0", upstreamType),
							upstreamType,
							inputs,
							inputGradExprs,
							inputGradTypes);
					} else {
						CollectExpressionLeaves(*param, SwizzleComponentExpr(upstream, static_cast<size_t>(componentOffset)),
												"float", inputs, inputGradExprs, inputGradTypes);
					}
				} else {
					std::ostringstream coeff;
					coeff << argType << "(";
					for (int i = 0; i < take; i++) {
						if (i > 0) {
							coeff << ", ";
						}
						coeff << SwizzleComponentExpr(upstream, static_cast<size_t>(componentOffset + i));
					}
					coeff << ")";
					CollectExpressionLeaves(*param, coeff.str(), argType, inputs, inputGradExprs, inputGradTypes);
				}

				componentOffset += take;
			}
			return;
		}

		if (params.size() == 1 && params[0]) {
			std::string argExpr = nodeExpr(*params[0]);
			if (intrinsicName == "normalize") {
				std::string xn = std::format("({})/length({})", argExpr, argExpr);
				std::string proj = std::format("({})*dot({},{})", xn, xn, upstream);
				std::string tangent = std::format("({})-({})", upstream, proj);
				CollectExpressionLeaves(*params[0], std::format("({})/length({})", tangent, argExpr),
										InferNodeType(*params[0]), inputs, inputGradExprs, inputGradTypes);
				return;
			}
			if (intrinsicName == "length") {
				CollectExpressionLeaves(*params[0], std::format("({})*({})/length({})", upstream, argExpr, argExpr),
										InferNodeType(*params[0]), inputs, inputGradExprs, inputGradTypes);
				return;
			}

			std::string deriv	= SingleArgumentIntrinsicDerivative(intrinsicName, argExpr);
			if (!deriv.empty()) {
				CollectExpressionLeaves(*params[0], std::format("({})*({})", upstream, deriv), upstreamType, inputs,
										inputGradExprs, inputGradTypes);
			}
		}
		return;
	}

	if (node.Type() != GPU::IR::Node::NodeType::Operation) {
		AddExpressionLeaf(node, upstream, upstreamType, inputs, inputGradExprs, inputGradTypes);
		return;
	}

	const auto &opNode = static_cast<const GPU::IR::Node::OperationNode &>(node);
	const auto *lhs	   = opNode.LHS();
	const auto *rhs	   = opNode.RHS();
	if (!lhs)
		return;

	switch (opNode.Code()) {
	case GPU::IR::Node::OperationCode::Add:
		if (rhs) {
			CollectExpressionLeaves(*lhs, upstream, upstreamType, inputs, inputGradExprs, inputGradTypes);
			CollectExpressionLeaves(*rhs, upstream, upstreamType, inputs, inputGradExprs, inputGradTypes);
		}
		break;
	case GPU::IR::Node::OperationCode::Sub:
		if (rhs) {
			CollectExpressionLeaves(*lhs, upstream, upstreamType, inputs, inputGradExprs, inputGradTypes);
			CollectExpressionLeaves(*rhs, std::format("-({})", upstream), upstreamType, inputs, inputGradExprs,
									inputGradTypes);
		}
		break;
	case GPU::IR::Node::OperationCode::Mul:
		if (rhs) {
			std::string lhsExpr = nodeExpr(*lhs);
			std::string rhsExpr = nodeExpr(*rhs);
			CollectExpressionLeaves(*lhs, std::format("({})*({})", upstream, rhsExpr), upstreamType, inputs,
									inputGradExprs, inputGradTypes);
			CollectExpressionLeaves(*rhs, std::format("({})*({})", upstream, lhsExpr), upstreamType, inputs,
									inputGradExprs, inputGradTypes);
		}
		break;
	case GPU::IR::Node::OperationCode::Div:
		if (rhs) {
			std::string lhsExpr = nodeExpr(*lhs);
			std::string rhsExpr = nodeExpr(*rhs);
			CollectExpressionLeaves(*lhs, std::format("({})/({})", upstream, rhsExpr), upstreamType, inputs,
									inputGradExprs, inputGradTypes);
			CollectExpressionLeaves(*rhs, std::format("-(({})*({})/(({})*({})))", upstream, lhsExpr, rhsExpr, rhsExpr),
									upstreamType, inputs, inputGradExprs, inputGradTypes);
		}
		break;
	case GPU::IR::Node::OperationCode::Neg:
		CollectExpressionLeaves(*lhs, std::format("-({})", upstream), upstreamType, inputs, inputGradExprs,
								inputGradTypes);
		break;
	default:
		break;
	}
}

// =============================================================================
// Helpers
// =============================================================================

std::string GradientTape::ExtractVarName(const GPU::IR::Node::Node &loadNode) {
	if (loadNode.Type() == GPU::IR::Node::NodeType::Load) {
		return static_cast<const GPU::IR::Node::LoadNode &>(loadNode).Unwrap();
	}
	if (loadNode.Type() == GPU::IR::Node::NodeType::ArrayAccess) {
		const auto &array = static_cast<const GPU::IR::Node::ArrayAccessNode &>(loadNode);
		if (array.Target() == nullptr || array.Index() == nullptr)
			return "";
		auto target = ExtractVarName(*array.Target());
		if (target.empty())
			return "";
		auto index = GPU::IR::Builder::Builder::Get().BuildNode(*array.Index());
		if (index.empty())
			return "";
		return target + "[" + index + "]";
	}
	return "";
}

std::string GradientTape::TryExtractVarName(const GPU::IR::Node::Node &node) {
	if (node.Type() == GPU::IR::Node::NodeType::Load) {
		auto	   &loadNode = static_cast<const GPU::IR::Node::LoadNode &>(node);
		std::string s		 = loadNode.Unwrap();
		if (s.empty())
			return "";
		// Filter GLSL type constructors: float(...), vec2(...), etc.
		if (s.find('(') != std::string::npos)
			return "";
		// Filter boolean literals
		if (s == "true" || s == "false")
			return "";
		return s;
	}
	return "";
}

std::string GradientTape::InferNodeType(const GPU::IR::Node::Node &node) const {
	std::string existing = TryExtractVarName(node);
	if (existing.empty()) {
		existing = ExtractVarName(node);
	}
	if (!existing.empty()) {
		if (auto *tp = GetVarType(existing)) {
			return *tp;
		}
	}

	switch (node.Type()) {
	case GPU::IR::Node::NodeType::Operation: {
		const auto &operation = static_cast<const GPU::IR::Node::OperationNode &>(node);
		if (operation.LHS() != nullptr) {
			return InferNodeType(*operation.LHS());
		}
		break;
	}
	case GPU::IR::Node::NodeType::CallInst: {
		const auto &intrinsic = static_cast<const GPU::IR::Node::IntrinsicCallNode &>(node);
		const auto name = std::string(intrinsic.Name());
		if (name == "vec2" || name == "ivec2") {
			return name;
		}
		if (name == "vec3" || name == "ivec3") {
			return name;
		}
		if (name == "vec4" || name == "ivec4") {
			return name;
		}
		if (name == "normalize" && intrinsic.Parameter().size() == 1 && intrinsic.Parameter()[0] != nullptr) {
			return InferNodeType(*intrinsic.Parameter()[0]);
		}
		if (name == "length" || name == "dot" || name == "distance") {
			return "float";
		}
		break;
	}
	case GPU::IR::Node::NodeType::MemberAccess: {
		const auto &member = static_cast<const GPU::IR::Node::MemberAccessNode &>(node);
		if (member.RHS() != nullptr) {
			const auto memberName = GPU::IR::Builder::Builder::Get().BuildNode(*member.RHS());
			if (memberName.size() == 1 &&
				std::string_view("xyzwrgba").find(memberName[0]) != std::string_view::npos) {
				return "float";
			}
			if (memberName == "xy" || memberName == "rg") {
				return "vec2";
			}
			if (memberName == "xyz" || memberName == "rgb") {
				return "vec3";
			}
			if (memberName == "rgba") {
				return "vec4";
			}
		}
		break;
	}
	case GPU::IR::Node::NodeType::ArrayAccess: {
		const auto &array = static_cast<const GPU::IR::Node::ArrayAccessNode &>(node);
		if (array.Target() != nullptr) {
			return InferNodeType(*array.Target());
		}
		break;
	}
	default:
		break;
	}

	return "float";
}

std::string GradientTape::ExtractMemberName(const GPU::IR::Node::Node &memberNode) {
	std::string member = ExtractVarName(memberNode);
	if (member.empty()) {
		member = GPU::IR::Builder::Builder::Get().BuildNode(memberNode);
	}
	while (member.size() >= 2 && member.front() == '(' && member.back() == ')') {
		member = member.substr(1, member.size() - 2);
	}
	return member;
}

std::string GradientTape::BuildSwizzleScatterExpression(
	const std::string &memberName,
	const std::string &upstream,
	const std::string &baseType) {
	const int size = VectorSize(baseType);
	if (size <= 0) {
		return upstream;
	}

	std::vector<std::string> components(static_cast<size_t>(size), "0.0");
	for (size_t i = 0; i < memberName.size(); i++) {
		const int componentIndex = SwizzleComponentIndex(memberName[i]);
		if (componentIndex < 0 || componentIndex >= size) {
			continue;
		}

		std::string contribution;
		if (memberName.size() == 1) {
			contribution = upstream;
		} else {
			contribution = std::format("({}).{}", upstream, VectorComponentName(i));
		}

		auto &slot = components[static_cast<size_t>(componentIndex)];
		if (slot == "0.0") {
			slot = contribution;
		} else {
			slot = std::format("({})+({})", slot, contribution);
		}
	}

	std::ostringstream expr;
	expr << baseType << "(";
	for (size_t i = 0; i < components.size(); i++) {
		if (i > 0) {
			expr << ", ";
		}
		expr << components[i];
	}
	expr << ")";
	return expr.str();
}

TapeOpKind GradientTape::ClassifyOp(GPU::IR::Node::OperationCode code) {
	switch (code) {
	case GPU::IR::Node::OperationCode::Neg:
		return TapeOpKind::UnaryOp;
	default:
		return TapeOpKind::BinaryOp;
	}
}

void GradientTape::PropagateActive(const TapeVar &output, const std::vector<TapeVar> &inputs) {
	bool outActive		= IsActive(output.name) || IsParameter(output.name);
	bool anyInputActive = false;
	for (const auto &in : inputs) {
		if (IsActive(in.name) || IsParameter(in.name)) {
			anyInputActive = true;
			break;
		}
	}

	if (outActive || anyInputActive) {
		_activeNames.insert(output.name);
		for (const auto &in : inputs) {
			_activeNames.insert(in.name);
		}
	}
}

// =============================================================================
// Control flow markers
// =============================================================================

void GradientTape::BeginIfBranch(const std::string &conditionExpr) {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id			   = target->_nextId++;
	entry.kind			   = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind  = ControlFlowKind::IfBranch;
	entry.conditionVarName = conditionExpr;
	target->_entries.push_back(std::move(entry));
}

void GradientTape::BeginElifBranch(const std::string &conditionExpr) {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id			   = target->_nextId++;
	entry.kind			   = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind  = ControlFlowKind::ElifBranch;
	entry.conditionVarName = conditionExpr;
	target->_entries.push_back(std::move(entry));
}

void GradientTape::BeginElseBranch() {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id			  = target->_nextId++;
	entry.kind			  = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::ElseBranch;
	target->_entries.push_back(std::move(entry));
}

void GradientTape::EndIfChain() {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id   = target->_nextId++;
	entry.kind = TapeOpKind::ControlFlowEnd;
	target->_entries.push_back(std::move(entry));
}

void GradientTape::BeginForLoop(const std::string &varName, const std::string &start, const std::string &end,
								const std::string &step) {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id			  = target->_nextId++;
	entry.kind			  = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::ForLoop;
	entry.forVarName	  = varName;
	entry.forStart		  = start;
	entry.forEnd		  = end;
	entry.forStep		  = step;
	target->_entries.push_back(std::move(entry));
}

void GradientTape::EndForLoop() {
	auto *target = _currentSubTape != nullptr ? _currentSubTape : this;
	TapeEntry entry;
	entry.id   = target->_nextId++;
	entry.kind = TapeOpKind::ControlFlowEnd;
	target->_entries.push_back(std::move(entry));
}

bool GradientTape::IsTapeActive() {
	return ::GPU::IR::Builder::Builder::Get().GetGradientTape() != nullptr;
}

bool GradientTape::IsActive() {
	return IsTapeActive();
}

// =============================================================================
// Sub-tape support (Callable body recording)
// =============================================================================

void GradientTape::PushSubTape(const std::string &callableName) {
	auto sub			 = std::make_unique<GradientTape>();
	_currentSubTape		 = sub.get();
	// Push to the current active tape (this or a sub-tape) so the hierarchy
	// forms a proper tree. Otherwise nested Flow::For / Flow::If bodies would
	// all be flattened into the main tape's _subTapes and recursion in
	// ProcessCall (via CloneSubTapesFrom) would lose them.
	GradientTape *parent = _subTapeStack.empty() ? this : _subTapeStack.top();
	_subTapeParentStack.push(parent);
	_subTapeStack.push(_currentSubTape);
	auto index = static_cast<int>(parent->_subTapes.size());
	parent->_subTapes.push_back(std::move(sub));
	parent->_subTapeCallableNames.push_back(callableName);
	if (!callableName.empty()) {
		parent->_subTapeCallableNameToIndex[callableName] = index;
	}
}

int GradientTape::PopSubTape() {
	if (_subTapeStack.empty())
		return -1;
	GradientTape *current = _currentSubTape;
	GradientTape *parent  = _subTapeParentStack.empty() ? this : _subTapeParentStack.top();
	int			  index	  = -1;
	// Find the index of the current sub-tape
	for (int i = 0; i < (int)parent->_subTapes.size(); i++) {
		if (parent->_subTapes[i].get() == current) {
			index = i;
			break;
		}
	}
	_subTapeStack.pop();
	if (!_subTapeParentStack.empty()) {
		_subTapeParentStack.pop();
	}
	_currentSubTape = _subTapeStack.empty() ? nullptr : _subTapeStack.top();
	return index;
}

void GradientTape::CloneSubTapesFrom(const GradientTape &src) {
	for (size_t i = 0; i < src.SubTapeCount(); i++) {
		const auto &ss	 = src.SubTape(i);
		auto		copy = std::make_unique<GradientTape>();
		for (size_t j = 0; j < ss.Size(); j++) {
			copy->RecordRemapped(ss[j]);
		}
		copy->CloneSubTapesFrom(ss);
		std::string callableName;
		if (i < src._subTapeCallableNames.size()) {
			callableName = src._subTapeCallableNames[i];
		}
		auto index = static_cast<int>(_subTapes.size());
		_subTapes.push_back(std::move(copy));
		_subTapeCallableNames.push_back(callableName);
		if (!callableName.empty()) {
			_subTapeCallableNameToIndex[callableName] = index;
		}
	}
}

const GradientTape *GradientTape::FindSubTapeByCallableName(const std::string &callableName, int *index) const {
	if (callableName.empty()) {
		return nullptr;
	}

	auto it = _subTapeCallableNameToIndex.find(callableName);
	if (it == _subTapeCallableNameToIndex.end()) {
		return nullptr;
	}

	const auto subTapeIndex = it->second;
	if (subTapeIndex < 0 || static_cast<size_t>(subTapeIndex) >= _subTapes.size()) {
		return nullptr;
	}

	if (index != nullptr) {
		*index = subTapeIndex;
	}
	return _subTapes[subTapeIndex].get();
}

// =============================================================================
// Callable invocation recording
// =============================================================================

void GradientTape::RecordCall(const GPU::IR::Node::CallNode &callNode, const TapeVar &output) {
	std::vector<TapeVar> inputs;
	for (const auto &arg : callNode.Arguments()) {
		std::string n = TryExtractVarName(*arg);
		if (n.empty())
			n = ExtractVarName(*arg);
		std::string t;
		if (auto *tp = GetVarType(n))
			t = *tp;
		else
			t = InferNodeType(*arg);
		inputs.push_back(TapeVar{n, t, IsParameter(n)});
	}

	auto entry			   = MakeEntry(_nextId++, TapeOpKind::Call, output, inputs);
	entry.callableFuncName = callNode.FuncName();
	entry.forwardExpr	   = BuildNodeExpression(callNode);
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

// =============================================================================
// Return recording (callable body return value)
// =============================================================================

void GradientTape::RecordReturn(const GPU::IR::Node::ReturnNode &retNode) {
	// This is only meaningful inside a callable body (sub-tape).
	// Record which variable is returned so the adjoint generator knows
	// where to seed the adjoint.
	if (retNode.Value()) {
		std::string name = TryExtractVarName(*retNode.Value());
		if (name.empty())
			name = ExtractVarName(*retNode.Value());
		if (name.empty()) {
			name = std::format("_return{}", _nextId);
			const auto returnType = InferNodeType(*retNode.Value());
			TapeVar output{name, returnType, false};
			_varTypes[name] = returnType;
			RecordAssignmentRhs(*retNode.Value(), output);
		}
		if (!name.empty()) {
			std::string t;
			if (auto *tp = GetVarType(name))
				t = *tp;
			else
				t = "float";

			auto entry = MakeEntry(_nextId++, TapeOpKind::Return, TapeVar{name, t, IsParameter(name)}, {});
			_entries.push_back(std::move(entry));
		}
	} else {
		// void return
		auto entry = MakeEntry(_nextId++, TapeOpKind::Return, TapeVar{"", "void", false}, {});
		_entries.push_back(std::move(entry));
	}
}

} // namespace GPU::AD
