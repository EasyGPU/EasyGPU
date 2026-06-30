/**
 * @file Builder.cpp
 * @brief Implementation of the IR builder for constructing GPU shader code from DSL expressions.
 */

#include <IR/Builder/Builder.h>

#include <AD/GradientTape.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/AtomicOp.h>
#include <IR/Node/Barrier.h>
#include <IR/Node/Break.h>
#include <IR/Node/Call.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/CompoundAssignment.h>
#include <IR/Node/Continue.h>
#include <IR/Node/DoWhile.h>
#include <IR/Node/For.h>
#include <IR/Node/If.h>
#include <IR/Node/Increment.h>
#include <IR/Node/Load.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/LocalVariableArray.h>
#include <IR/Node/MemberAccess.h>
#include <IR/Node/Node.h>
#include <IR/Node/Operation.h>
#include <IR/Node/RawCode.h>
#include <IR/Node/Return.h>
#include <IR/Node/SharedMemory.h>
#include <IR/Node/Store.h>
#include <IR/Node/Ternary.h>
#include <IR/Node/TextureLoad.h>
#include <IR/Node/TextureSample.h>
#include <IR/Node/TextureStore.h>
#include <IR/Node/While.h>

#include <format>
#include <sstream>

namespace GPU::IR::Builder {
namespace {
bool IsLegitimateEmptyBuild(const Node::Node &node) {
	if (node.Type() == Node::NodeType::SharedMemory) {
		return true;
	}
	if (node.Type() == Node::NodeType::Barrier) {
		return false;
	}
	if (node.Type() == Node::NodeType::LocalVariable) {
		const auto &local = static_cast<const Node::LocalVariableNode &>(node);
		return local.IsExternal();
	}
	return false;
}

bool NeedsStatementTerminator(const Node::Node &node) {
	switch (node.Type()) {
	case Node::NodeType::Store:
	case Node::NodeType::CompoundAssignment:
	case Node::NodeType::Increment:
	case Node::NodeType::Break:
	case Node::NodeType::Continue:
	case Node::NodeType::Return:
	case Node::NodeType::Call:
	case Node::NodeType::CallInst:
	case Node::NodeType::LocalVariable:
	case Node::NodeType::LocalArray:
	case Node::NodeType::AtomicOp:
	case Node::NodeType::Barrier:
	case Node::NodeType::TextureStore:
	case Node::NodeType::RawCode:
		return true;
	default:
		return false;
	}
}

void AppendStatementCode(std::string &code, const Node::Node &node, Builder &builder) {
	const auto statementCode = builder.BuildNode(node);
	if (statementCode.empty()) {
		if (IsLegitimateEmptyBuild(node)) {
			return;
		}
		builder.ValidateGeneratedCode(statementCode, "statement");
	}

	code.append(statementCode);
	if (NeedsStatementTerminator(node)) {
		code.append(";");
	}
	code.append("\n");
}

std::string BuildForHeaderNodes(const std::vector<std::unique_ptr<Node::Node>> &nodes, Builder &builder) {
	std::string code;
	for (const auto &node : nodes) {
		if (!node) {
			continue;
		}

		const auto part = builder.BuildNode(*node);
		if (part.empty()) {
			if (IsLegitimateEmptyBuild(*node)) {
				continue;
			}
			builder.ValidateGeneratedCode(part, "for header");
		}

		if (!code.empty()) {
			code.append(", ");
		}
		code.append(part);
	}
	return code;
}
} // namespace

Builder &Builder::Get() {
	thread_local static Builder builder;

	return builder;
}

void Builder::Bind(BuilderContext &Context) {
	// Push current context to stack before binding new one (support nested definitions)
	if (_context != nullptr) {
		if (_contextStack.size() >= kMaxContextStackDepth) {
			throw std::runtime_error("EasyGPU builder context stack exceeded maximum nesting depth");
		}
		_contextStack.push(_context);
	}
	_context = &Context;
}

void Builder::Unbind() {
	if (_context == nullptr) {
		throw std::runtime_error("EasyGPU builder Unbind() called with no active context");
	}
	// Restore previous context from stack if available
	if (!_contextStack.empty()) {
		_context = _contextStack.top();
		_contextStack.pop();
	} else {
		_context = nullptr;
	}
}

BuilderContext *Builder::Context() {
	return _context;
}

BuilderContext *Builder::ContextChecked() {
	if (_context == nullptr) {
		throw std::runtime_error("EasyGPU DSL operation called outside of Kernel definition");
	}
	return _context;
}

void Builder::Build(const Node::Node &Node, bool IsStatement) {
	if (_context != nullptr) {
		std::string code = BuildNode(Node);
		if (code.empty() && IsLegitimateEmptyBuild(Node)) {
			return;
		}
		ValidateGeneratedCode(code, IsStatement ? "statement" : "expression");
		if (IsStatement) {
			if (NeedsStatementTerminator(Node)) {
				code.append(";");
			}
			code.append("\n");
			_context->PushTranslatedCode(code);
			if (_gradientTape) {
				_gradientTape->Record(Node, IsStatement);
			}
		} else {
			_context->PushTranslatedCode(code);
		}
	}
}

std::string Builder::BuildNode(const Node::Node &Node) {
	switch (Node.Type()) {
	case Node::NodeType::CallInst: {
		return BuildCallInst(static_cast<const Node::IntrinsicCallNode &>(Node));
	}
	case Node::NodeType::Operation: {
		return BuildOperation(static_cast<const Node::OperationNode &>(Node));
	}
	case Node::NodeType::LocalVariable: {
		return BuildLocalVariable(static_cast<const Node::LocalVariableNode &>(Node));
	}
	case Node::NodeType::Load: {
		return BuildLoad(static_cast<const Node::LoadNode &>(Node));
	}
	case Node::NodeType::Store: {
		return BuildStore(static_cast<const Node::StoreNode &>(Node));
	}
	case Node::NodeType::LocalArray: {
		return BuildLocalVariableArray(static_cast<const Node::LocalVariableArrayNode &>(Node));
	}
	case Node::NodeType::ArrayAccess: {
		return BuildArrayAccess(static_cast<const Node::ArrayAccessNode &>(Node));
	}
	case Node::NodeType::CompoundAssignment: {
		return BuildCompoundAssignment(static_cast<const Node::CompoundAssignmentNode &>(Node));
	}
	case Node::NodeType::Increment: {
		return BuildIncrement(static_cast<const Node::IncrementNode &>(Node));
	}
	case Node::NodeType::MemberAccess: {
		return BuildMemberAccess(static_cast<const Node::MemberAccessNode &>(Node));
	}
	case Node::NodeType::If: {
		return BuildIf(static_cast<const Node::IfNode &>(Node));
	}
	case Node::NodeType::While: {
		return BuildWhile(static_cast<const Node::WhileNode &>(Node));
	}
	case Node::NodeType::DoWhile: {
		return BuildDoWhile(static_cast<const Node::DoWhileNode &>(Node));
	}
	case Node::NodeType::For: {
		return BuildFor(static_cast<const Node::ForNode &>(Node));
	}
	case Node::NodeType::TextureLoad: {
		return BuildTextureLoad(static_cast<const Node::TextureLoadNode &>(Node));
	}
	case Node::NodeType::TextureStore: {
		return BuildTextureStore(static_cast<const Node::TextureStoreNode &>(Node));
	}
	case Node::NodeType::TextureSample: {
		return BuildTextureSample(static_cast<const Node::TextureSampleNode &>(Node));
	}
	case Node::NodeType::Break: {
		return BuildBreak(static_cast<const Node::BreakNode &>(Node));
	}
	case Node::NodeType::Continue: {
		return BuildContinue(static_cast<const Node::ContinueNode &>(Node));
	}
	case Node::NodeType::Return: {
		return BuildReturn(static_cast<const Node::ReturnNode &>(Node));
	}
	case Node::NodeType::Call: {
		return BuildCall(static_cast<const Node::CallNode &>(Node));
	}
	case Node::NodeType::RawCode: {
		return BuildRawCode(static_cast<const Node::RawCodeNode &>(Node));
	}
	case Node::NodeType::Ternary: {
		return BuildTernary(static_cast<const Node::TernaryNode &>(Node));
	}
	case Node::NodeType::SharedMemory: {
		return BuildSharedMemory(static_cast<const Node::SharedMemoryNode &>(Node));
	}
	case Node::NodeType::AtomicOp: {
		return BuildAtomicOp(static_cast<const Node::AtomicOpNode &>(Node));
	}
	case Node::NodeType::Barrier: {
		return BuildBarrier(static_cast<const Node::BarrierNode &>(Node));
	}
	default: {
		return "";
	}
	}
}

std::string Builder::BuildCallInst(const Node::IntrinsicCallNode &Node) {
	std::ostringstream stream;

	stream << Node.Name() << "(";
	if (!Node.Parameter().empty()) {
		stream << BuildNode(*Node.Parameter()[0]);
		for (size_t index = 1; index < Node.Parameter().size(); ++index) {
			stream << "," << BuildNode(*Node.Parameter()[index]);
		}
	}
	stream << ")";

	return stream.str();
}

std::string Builder::BuildOperation(const Node::OperationNode &Node) {
	switch (Node.Code()) {
	case Node::OperationCode::Add: {
		return std::format("({})+({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Sub: {
		return std::format("({})-({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Mul: {
		return std::format("({})*({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Div: {
		return std::format("({})/({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Mod: {
		return std::format("({})%({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Neg: {
		return std::format("-({})", BuildNode(*Node.LHS()));
	}
	case Node::OperationCode::BitAnd: {
		return std::format("({})&({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::BitOr: {
		return std::format("({})|({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::BitXor: {
		return std::format("({})^({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::BitNot: {
		return std::format("~({})", BuildNode(*Node.LHS()));
	}
	case Node::OperationCode::Shl: {
		return std::format("({})<<({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Shr: {
		return std::format("({})>>({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Less: {
		return std::format("({})<({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Greater: {
		return std::format("({})>({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::Equal: {
		return std::format("({})==({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::NotEqual: {
		return std::format("({})!=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::LessEqual: {
		return std::format("({})<=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::GreaterEqual: {
		return std::format("({})>=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::LogicalAnd: {
		return std::format("({})&&({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::LogicalOr: {
		return std::format("({})||({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::OperationCode::LogicalNot: {
		return std::format("!({})", BuildNode(*Node.LHS()));
	}
	default: {
		return "";
	}
	}
}

std::string Builder::BuildLocalVariable(const Node::LocalVariableNode &Node) {
	// External variables (e.g., uniforms) are declared outside main(),
	// so we don't need to declare them in the main function body
	if (Node.IsExternal()) {
		return "";
	}
	if (Node.HasInitializer()) {
		return std::format("{} {} = {}", Node.VarType(), Node.VarName(), BuildNode(*Node.Initializer()));
	}
	return std::format("{} {}", Node.VarType(), Node.VarName());
}

std::string Builder::BuildLocalVariableArray(const Node::LocalVariableArrayNode &Node) {
	return std::format("{} {}[{}]", Node.VarType(), Node.VarName(), Node.Size());
}

std::string Builder::BuildLoad(const Node::LoadNode &Node) {
	return Node.Unwrap();
}

std::string Builder::BuildStore(const Node::StoreNode &Node) {
	return std::format("({})=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
}

std::string Builder::BuildArrayAccess(const Node::ArrayAccessNode &Node) {
	return std::format("({})[{}]", BuildNode(*Node.Target()), BuildNode(*Node.Index()));
}

std::string Builder::BuildCompoundAssignment(const Node::CompoundAssignmentNode &Node) {
	switch (Node.Code()) {
	case Node::CompoundAssignmentCode::AddAssign: {
		return std::format("({}) += ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::SubAssign: {
		return std::format("({}) -= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::MulAssign: {
		return std::format("({}) *= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::DivAssign: {
		return std::format("({}) /= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::ModAssign: {
		return std::format("({}) %= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::BitAndAssign: {
		return std::format("({}) &= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::BitOrAssign: {
		return std::format("({}) |= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::BitXorAssign: {
		return std::format("({}) ^= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::ShlAssign: {
		return std::format("({}) <<= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	case Node::CompoundAssignmentCode::ShrAssign: {
		return std::format("({}) >>= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
	}
	default: {
		return "";
	}
	}
}

std::string Builder::BuildIncrement(const Node::IncrementNode &Node) {
	switch (Node.Direction()) {
	case Node::IncrementDirection::Decrement: {
		if (Node.IsPrefix()) {
			return std::format("--({})", BuildNode(*Node.Target()));
		} else {
			return std::format("({})--", BuildNode(*Node.Target()));
		}
	}
	case Node::IncrementDirection::Increment: {
		if (Node.IsPrefix()) {
			return std::format("++({})", BuildNode(*Node.Target()));
		} else {
			return std::format("({})++", BuildNode(*Node.Target()));
		}
	}
	default: {
		return "";
	}
	}
}

std::string Builder::BuildMemberAccess(const Node::MemberAccessNode &Node) {
	return std::format("({}).{}", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
}

std::string Builder::BuildIf(const Node::IfNode &Node) {
	std::string code = std::format("if ({}) {{\n", BuildNode(*Node.Condition()));
	for (auto &node : Node.Do()) {
		AppendStatementCode(code, *node, *this);
	}
	code.append("}");

	// Build elif branches
	for (const auto &[elifCond, elifBody] : Node.Elifs()) {
		code.append(std::format(" else if ({}) {{\n", BuildNode(*elifCond)));
		for (auto &node : elifBody) {
			AppendStatementCode(code, *node, *this);
		}
		code.append("}");
	}

	// Build else branch
	if (!Node.Else().empty()) {
		code.append(" else {\n");
		for (auto &node : Node.Else()) {
			AppendStatementCode(code, *node, *this);
		}
		code.append("}");
	}

	return code;
}

std::string Builder::BuildWhile(const Node::WhileNode &Node) {
	std::string code = std::format("while ({}) {{\n", BuildNode(*Node.Condition()));
	for (auto &node : Node.Body()) {
		AppendStatementCode(code, *node, *this);
	}
	code.append("}");
	return code;
}

std::string Builder::BuildDoWhile(const Node::DoWhileNode &Node) {
	std::string code = "do {\n";
	for (auto &node : Node.Body()) {
		AppendStatementCode(code, *node, *this);
	}
	code.append(std::format("}} while ({});", BuildNode(*Node.Condition())));
	return code;
}

std::string Builder::BuildFor(const Node::ForNode &Node) {
	if (Node.HasDynamicHeader()) {
		const auto condition = Node.Condition() ? BuildNode(*Node.Condition()) : "true";
		std::string code = std::format("for ({}; {}; {}) {{\n",
									   BuildForHeaderNodes(Node.Init(), *this),
									   condition,
									   BuildForHeaderNodes(Node.StepNodes(), *this));
		for (auto &node : Node.Body()) {
			AppendStatementCode(code, *node, *this);
		}
		code.append("}");
		return code;
	}

	std::string code = std::format("for (int {} = {}; {} < {}; {} += {}) {{\n", Node.VarName(), Node.Start(),
								   Node.VarName(), Node.End(), Node.VarName(), Node.Step());
	for (auto &node : Node.Body()) {
		AppendStatementCode(code, *node, *this);
	}
	code.append("}");
	return code;
}

std::string Builder::BuildBreak(const Node::BreakNode &Node) {
	return "break";
}

std::string Builder::BuildContinue(const Node::ContinueNode &Node) {
	return "continue";
}

std::string Builder::BuildReturn(const Node::ReturnNode &Node) {
	if (Node.HasValue()) {
		return std::format("return {}", BuildNode(*Node.Value()));
	}
	return "return";
}

std::string Builder::BuildCall(const Node::CallNode &Node) {
	std::ostringstream stream;
	stream << Node.FuncName() << "(";
	const auto &args = Node.Arguments();
	if (!args.empty()) {
		stream << BuildNode(*args[0]);
		for (size_t i = 1; i < args.size(); ++i) {
			stream << ", " << BuildNode(*args[i]);
		}
	}
	stream << ")";
	return stream.str();
}

std::string Builder::BuildRawCode(const Node::RawCodeNode &Node) {
	return Node.Code();
}

std::string Builder::BuildTernary(const Node::TernaryNode &Node) {
	return std::format("({})?({}):({})", BuildNode(*Node.Condition()), BuildNode(*Node.TrueExpr()),
					   BuildNode(*Node.FalseExpr()));
}

std::string Builder::BuildTextureLoad(const Node::TextureLoadNode &Node) {
	if (Node.X() == nullptr || Node.Y() == nullptr) {
		return "";
	}
	if (Node.Z() != nullptr) {
		return std::format("imageLoad({}, ivec3({}, {}, {}))", Node.TextureName(), BuildNode(*Node.X()),
						   BuildNode(*Node.Y()), BuildNode(*Node.Z()));
	}
	return std::format("imageLoad({}, ivec2({}, {}))", Node.TextureName(), BuildNode(*Node.X()), BuildNode(*Node.Y()));
}

std::string Builder::BuildTextureStore(const Node::TextureStoreNode &Node) {
	if (Node.X() == nullptr || Node.Y() == nullptr || Node.Value() == nullptr) {
		return "";
	}
	if (Node.Z() != nullptr) {
		return std::format("imageStore({}, ivec3({}, {}, {}), {})", Node.TextureName(), BuildNode(*Node.X()),
						   BuildNode(*Node.Y()), BuildNode(*Node.Z()), BuildNode(*Node.Value()));
	}
	return std::format("imageStore({}, ivec2({}, {}), {})", Node.TextureName(), BuildNode(*Node.X()),
					   BuildNode(*Node.Y()), BuildNode(*Node.Value()));
}

std::string Builder::BuildTextureSample(const Node::TextureSampleNode &Node) {
	if (Node.Coordinate() == nullptr) {
		return "";
	}
	if (Node.HasExplicitLevel()) {
		if (Node.Level() == nullptr) {
			return "";
		}
		return std::format("textureLod({}, {}, {})", Node.TextureName(), BuildNode(*Node.Coordinate()),
						   BuildNode(*Node.Level()));
	}
	return std::format("texture({}, {})", Node.TextureName(), BuildNode(*Node.Coordinate()));
}

std::string Builder::BuildSharedMemory(const Node::SharedMemoryNode &Node) {
	ContextChecked()->PushSharedMemoryDeclaration(
		std::format("shared {} {}[{}];", Node.VarType(), Node.VarName(), Node.Size()));
	return "";
}

std::string Builder::BuildAtomicOp(const Node::AtomicOpNode &Node) {
	std::string opName;
	switch (Node.Code()) {
	case Node::AtomicOpCode::Add:
		opName = "atomicAdd";
		break;
	case Node::AtomicOpCode::Sub:
		if (Node.Value() == nullptr) {
			return "";
		}
		return std::format("atomicAdd({}, -({}))", BuildNode(*Node.Target()), BuildNode(*Node.Value()));
	case Node::AtomicOpCode::Min:
		opName = "atomicMin";
		break;
	case Node::AtomicOpCode::Max:
		opName = "atomicMax";
		break;
	case Node::AtomicOpCode::And:
		opName = "atomicAnd";
		break;
	case Node::AtomicOpCode::Or:
		opName = "atomicOr";
		break;
	case Node::AtomicOpCode::Xor:
		opName = "atomicXor";
		break;
	case Node::AtomicOpCode::Exchange:
		opName = "atomicExchange";
		break;
	case Node::AtomicOpCode::CompSwap:
		opName = "atomicCompSwap";
		break;
	default:
		return "";
	}

	if (Node.IsCompSwap()) {
		return std::format("{}({}, {}, {})", opName, BuildNode(*Node.Target()), BuildNode(*Node.Compare()),
						   BuildNode(*Node.Value()));
	}
	return std::format("{}({}, {})", opName, BuildNode(*Node.Target()), BuildNode(*Node.Value()));
}

std::string Builder::BuildBarrier(const Node::BarrierNode &Node) {
	switch (Node.Code()) {
	case Node::BarrierCode::Workgroup:
		return "barrier()";
	case Node::BarrierCode::Memory:
		return "memoryBarrier()";
	case Node::BarrierCode::Full:
		return "memoryBarrier();\nbarrier()";
	default:
		return "";
	}
}

} // namespace GPU::IR::Builder
