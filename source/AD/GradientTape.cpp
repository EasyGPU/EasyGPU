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
#include <IR/Node/Load.h>
#include <IR/Node/LoadLocalArray.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/Node.h>
#include <IR/Node/Operation.h>
#include <IR/Node/Return.h>
#include <IR/Node/Store.h>
#include <IR/Node/Ternary.h>

namespace GPU::AD {

// =============================================================================
// Helper: construct a TapeEntry by setting fields explicitly
// =============================================================================

namespace {

TapeEntry MakeEntry(int32_t id, TapeOpKind kind, const TapeVar &output,
					const std::vector<TapeVar> &inputs) {
	TapeEntry e;
	e.id = id;
	e.kind = kind;
	e.output = output;
	e.inputs = inputs;
	return e;
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
	e.id = _nextId++;
	_entries.push_back(std::move(e));
}

void GradientTape::RegisterParameter(const std::string &name, const std::string &glslType) {
	_parameters[name] = glslType;
	_activeNames.insert(name);
	if (!_varTypes.count(name)) {
		_varTypes[name] = glslType;
	}
}

bool GradientTape::IsParameter(const std::string &name) const {
	return _parameters.count(name) > 0;
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
	const auto *lhs = store.LHS();
	const auto *rhs = store.RHS();

	std::string outName = ExtractVarName(*lhs);
	std::string outType;
	if (auto *tp = GetVarType(outName)) outType = *tp;
	else outType = "float";

	TapeVar output{outName, outType, IsParameter(outName)};

	switch (rhs->Type()) {
	case GPU::IR::Node::NodeType::Operation:
		RecordOperation(static_cast<const GPU::IR::Node::OperationNode &>(*rhs), output);
		break;

	case GPU::IR::Node::NodeType::CallInst:
		RecordIntrinsic(static_cast<const GPU::IR::Node::IntrinsicCallNode &>(*rhs), output);
		break;

	case GPU::IR::Node::NodeType::Ternary:
		RecordTernary(static_cast<const GPU::IR::Node::TernaryNode &>(*rhs), output);
		break;

	case GPU::IR::Node::NodeType::Load: {
		// Only record if the RHS is an actual variable load, not a literal/constant.
		std::string inName = TryExtractVarName(*rhs);
		if (!inName.empty() && inName != outName) {
			std::string inType;
			if (auto *t = GetVarType(inName)) inType = *t;
			else inType = "float";

			auto entry = MakeEntry(_nextId++, TapeOpKind::BinaryOp, output,
								   {TapeVar{inName, inType, IsParameter(inName)},
									TapeVar{"0", "float", false}});
			entry.binaryOp = GPU::IR::Node::OperationCode::Add;
			_entries.push_back(std::move(entry));

			PropagateActive(output, {TapeVar{inName, inType, IsParameter(inName)}});
		}
		// If TryExtractVarName returned empty, it's a literal/uniform value
		// (e.g., `a = 2.0f;` where RHS is LoadUniform("float(2.0)")).
		// These have zero gradient and don't need tape entries.
		break;
	}

	case GPU::IR::Node::NodeType::Call: {
		RecordCall(static_cast<const GPU::IR::Node::CallNode &>(*rhs), output);
		break;
	}

	case GPU::IR::Node::NodeType::ArrayAccess:
	case GPU::IR::Node::NodeType::MemberAccess:
	default:
		break;
	}
}

void GradientTape::RecordCompoundAssignment(const GPU::IR::Node::CompoundAssignmentNode &node) {
	std::string name = ExtractVarName(*node.LHS());
	std::string rhsName = TryExtractVarName(*node.RHS());
	std::string type;
	if (auto *t = GetVarType(name)) type = *t;
	else type = "float";

	auto code = node.Code();
	if (code != GPU::IR::Node::CompoundAssignmentCode::AddAssign &&
		code != GPU::IR::Node::CompoundAssignmentCode::SubAssign) {
		return;
	}

	std::string rhsType;
	if (auto *t = GetVarType(rhsName)) rhsType = *t;
	else rhsType = "float";

	TapeVar output{name, type, IsParameter(name)};
	TapeVar rhsVar{rhsName, rhsType, IsParameter(rhsName)};

	auto entry = MakeEntry(_nextId++, TapeOpKind::CompoundAssign, output, {output, rhsVar});
	entry.compoundOp = code;
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

	TapeOpKind kind = ClassifyOp(code);
	std::vector<TapeVar> inputs;

	auto makeInput = [this](const GPU::IR::Node::Node &operand) -> TapeVar {
		std::string n = TryExtractVarName(operand);
		if (n.empty()) n = ExtractVarName(operand);
		std::string t;
		if (auto *tp = GetVarType(n)) t = *tp;
		else t = "float";
		return TapeVar{n, t, IsParameter(n)};
	};

	inputs.push_back(makeInput(*op.LHS()));
	if (op.RHS() != nullptr) {
		inputs.push_back(makeInput(*op.RHS()));
	}

	auto entry = MakeEntry(_nextId++, kind, output, inputs);
	entry.binaryOp = code;
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

void GradientTape::RecordIntrinsic(const GPU::IR::Node::IntrinsicCallNode &node,
									const TapeVar					&output) {
	const auto	&params = node.Parameter();
	size_t		 nParams = params.size();
	std::string  intrinsicName(node.Name());

	TapeOpKind kind;
	switch (nParams) {
	case 1:	 kind = TapeOpKind::Intrinsic1; break;
	case 2:	 kind = TapeOpKind::Intrinsic2; break;
	default: kind = TapeOpKind::Intrinsic3; break;
	}

	std::vector<TapeVar> inputs;
	for (const auto &p : params) {
		std::string n = ExtractVarName(*p);
		std::string t;
		if (auto *tp = GetVarType(n)) t = *tp;
		else t = "float";
		inputs.push_back(TapeVar{n, t, IsParameter(n)});
	}

	auto entry = MakeEntry(_nextId++, kind, output, inputs);
	entry.intrinsicName = intrinsicName;
	_entries.push_back(std::move(entry));

	PropagateActive(output, inputs);
}

void GradientTape::RecordTernary(const GPU::IR::Node::TernaryNode &node, const TapeVar &output) {
	auto makeInput = [this](const GPU::IR::Node::Node &operand) -> TapeVar {
		std::string n = ExtractVarName(operand);
		std::string t;
		if (auto *tp = GetVarType(n)) t = *tp;
		else t = "float";
		return TapeVar{n, t, IsParameter(n)};
	};

	std::vector<TapeVar> inputs;
	inputs.push_back(makeInput(*node.Condition()));
	inputs.push_back(makeInput(*node.TrueExpr()));
	inputs.push_back(makeInput(*node.FalseExpr()));

	auto entry = MakeEntry(_nextId++, TapeOpKind::Ternary, output, inputs);
	_entries.push_back(std::move(entry));

	if (IsActive(output.name) || IsParameter(output.name)) {
		_activeNames.insert(output.name);
		_activeNames.insert(inputs[1].name);
		_activeNames.insert(inputs[2].name);
	}
}

void GradientTape::RecordLocalVariable(const GPU::IR::Node::LocalVariableNode &node) {
	std::string name = node.VarName();
	if (node.IsExternal()) return;
	if (name.find("gl_GlobalInvocationID") != std::string::npos) return;

	_varTypes[name] = node.VarType();
}

// =============================================================================
// Helpers
// =============================================================================

std::string GradientTape::ExtractVarName(const GPU::IR::Node::Node &loadNode) {
	if (loadNode.Type() == GPU::IR::Node::NodeType::Load) {
		return static_cast<const GPU::IR::Node::LoadNode &>(loadNode).Unwrap();
	}
	return "";
}

std::string GradientTape::TryExtractVarName(const GPU::IR::Node::Node &node) {
	if (node.Type() == GPU::IR::Node::NodeType::Load) {
		auto	   &loadNode = static_cast<const GPU::IR::Node::LoadNode &>(node);
		std::string s = loadNode.Unwrap();
		if (!s.empty() && s[0] != 'f' && s.find('(') == std::string::npos) {
			return s;
		}
	}
	return "";
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
	bool outActive = IsActive(output.name) || IsParameter(output.name);
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
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::IfBranch;
	entry.conditionVarName = conditionExpr;
	_entries.push_back(std::move(entry));
}

void GradientTape::BeginElifBranch(const std::string &conditionExpr) {
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::ElifBranch;
	entry.conditionVarName = conditionExpr;
	_entries.push_back(std::move(entry));
}

void GradientTape::BeginElseBranch() {
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::ElseBranch;
	_entries.push_back(std::move(entry));
}

void GradientTape::EndIfChain() {
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowEnd;
	_entries.push_back(std::move(entry));
}

void GradientTape::BeginForLoop(const std::string &varName, const std::string &start, const std::string &end, const std::string &step) {
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowBegin;
	entry.controlFlowKind = ControlFlowKind::ForLoop;
	entry.forVarName = varName;
	entry.forStart = start;
	entry.forEnd = end;
	entry.forStep = step;
	_entries.push_back(std::move(entry));
}

void GradientTape::EndForLoop() {
	TapeEntry entry;
	entry.id = _nextId++;
	entry.kind = TapeOpKind::ControlFlowEnd;
	_entries.push_back(std::move(entry));
}

bool GradientTape::IsActive() {
	return ::GPU::IR::Builder::Builder::Get().GetGradientTape() != nullptr;
}

// =============================================================================
// Sub-tape support (Callable body recording)
// =============================================================================

void GradientTape::PushSubTape() {
	auto sub = std::make_unique<GradientTape>();
	_currentSubTape = sub.get();
	_subTapeStack.push(_currentSubTape);
	_subTapes.push_back(std::move(sub));
}

int GradientTape::PopSubTape() {
	if (_subTapeStack.empty()) return -1;
	int index = (int)_subTapes.size(); // will be size-1 after push_back, but we already pushed
	// Find the index of the current sub-tape
	for (int i = 0; i < (int)_subTapes.size(); i++) {
		if (_subTapes[i].get() == _currentSubTape) {
			index = i;
			break;
		}
	}
	_subTapeStack.pop();
	_currentSubTape = _subTapeStack.empty() ? nullptr : _subTapeStack.top();
	return index;
}

// =============================================================================
// Callable invocation recording
// =============================================================================

void GradientTape::RecordCall(const GPU::IR::Node::CallNode &callNode, const TapeVar &output) {
	std::vector<TapeVar> inputs;
	for (const auto &arg : callNode.Arguments()) {
		std::string n = TryExtractVarName(*arg);
		if (n.empty()) n = ExtractVarName(*arg);
		std::string t;
		if (auto *tp = GetVarType(n)) t = *tp;
		else t = "float";
		inputs.push_back(TapeVar{n, t, IsParameter(n)});
	}

	auto entry = MakeEntry(_nextId++, TapeOpKind::Call, output, inputs);
	entry.callableFuncName = callNode.FuncName();
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
		if (name.empty()) name = ExtractVarName(*retNode.Value());
		if (!name.empty()) {
			std::string t;
			if (auto *tp = GetVarType(name)) t = *tp;
			else t = "float";

			auto entry = MakeEntry(_nextId++, TapeOpKind::Return,
								   TapeVar{name, t, IsParameter(name)}, {});
			_entries.push_back(std::move(entry));
		}
	} else {
		// void return
		auto entry = MakeEntry(_nextId++, TapeOpKind::Return,
							   TapeVar{"", "void", false}, {});
		_entries.push_back(std::move(entry));
	}
}

} // namespace GPU::AD
