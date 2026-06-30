/**
 * @file Module.cpp
 * @brief Language-neutral EasyGPU IR module lowering.
 */

#include <IR/Module.h>

#include <IR/Builder/Builder.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/AtomicOp.h>
#include <IR/Node/Barrier.h>
#include <IR/Node/Break.h>
#include <IR/Node/Call.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/Continue.h>
#include <IR/Node/DoWhile.h>
#include <IR/Node/For.h>
#include <IR/Node/If.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Node/MemberAccess.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/Node.h>
#include <IR/Node/Operation.h>
#include <IR/Node/RawCode.h>
#include <IR/Node/Return.h>
#include <IR/Node/SharedMemory.h>
#include <IR/Node/Store.h>
#include <IR/Node/TextureLoad.h>
#include <IR/Node/TextureSample.h>
#include <IR/Node/TextureStore.h>
#include <IR/Node/While.h>
#include <AD/GradientTape.h>
#include <Flow/CodeCollectContext.h>
#include <Kernel/KernelBuildContext.h>

#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace GPU::IR {
namespace {

[[nodiscard]] std::string ToGlslType(const Type &type) {
	switch (type.kind) {
	case Type::Kind::Void:
		return "void";
	case Type::Kind::Bool:
		return "bool";
	case Type::Kind::Int:
		return "int";
	case Type::Kind::UInt:
		return "uint";
	case Type::Kind::Float:
		return "float";
	case Type::Kind::Bool2:
		return "bvec2";
	case Type::Kind::Bool3:
		return "bvec3";
	case Type::Kind::Bool4:
		return "bvec4";
	case Type::Kind::Int2:
		return "ivec2";
	case Type::Kind::Int3:
		return "ivec3";
	case Type::Kind::Int4:
		return "ivec4";
	case Type::Kind::UInt2:
		return "uvec2";
	case Type::Kind::UInt3:
		return "uvec3";
	case Type::Kind::UInt4:
		return "uvec4";
	case Type::Kind::Float2:
		return "vec2";
	case Type::Kind::Float3:
		return "vec3";
	case Type::Kind::Float4:
		return "vec4";
	case Type::Kind::Float2x2:
		return "mat2";
	case Type::Kind::Float3x3:
		return "mat3";
	case Type::Kind::Float4x4:
		return "mat4";
	case Type::Kind::Struct:
		return type.typeName;
	default:
		return {};
	}
}

[[nodiscard]] int ToBackendBufferMode(ResourceAccess access) {
	switch (access) {
	case ResourceAccess::Read:
		return Backend::BUFFER_MODE_READ_ONLY;
	case ResourceAccess::Write:
		return Backend::BUFFER_MODE_WRITE_ONLY;
	default:
		return Backend::BUFFER_MODE_READ_WRITE;
	}
}

[[nodiscard]] Node::OperationCode ToNodeOperation(BinaryOp op) {
	switch (op) {
	case BinaryOp::Add:
		return Node::OperationCode::Add;
	case BinaryOp::Sub:
		return Node::OperationCode::Sub;
	case BinaryOp::Mul:
		return Node::OperationCode::Mul;
	case BinaryOp::Div:
		return Node::OperationCode::Div;
	case BinaryOp::Mod:
		return Node::OperationCode::Mod;
	case BinaryOp::BitAnd:
		return Node::OperationCode::BitAnd;
	case BinaryOp::BitOr:
		return Node::OperationCode::BitOr;
	case BinaryOp::BitXor:
		return Node::OperationCode::BitXor;
	case BinaryOp::ShiftLeft:
		return Node::OperationCode::Shl;
	case BinaryOp::ShiftRight:
		return Node::OperationCode::Shr;
	case BinaryOp::LogicalAnd:
		return Node::OperationCode::LogicalAnd;
	case BinaryOp::LogicalOr:
		return Node::OperationCode::LogicalOr;
	default:
		return Node::OperationCode::Add;
	}
}

[[nodiscard]] Node::OperationCode ToNodeOperation(UnaryOp op) {
	switch (op) {
	case UnaryOp::Negate:
		return Node::OperationCode::Neg;
	case UnaryOp::LogicalNot:
		return Node::OperationCode::LogicalNot;
	case UnaryOp::BitwiseNot:
		return Node::OperationCode::BitNot;
	default:
		return Node::OperationCode::Neg;
	}
}

[[nodiscard]] bool TryMapBarrierCode(BarrierKind kind, Node::BarrierCode &code) {
	switch (kind) {
	case BarrierKind::Workgroup:
		code = Node::BarrierCode::Workgroup;
		return true;
	case BarrierKind::Memory:
		code = Node::BarrierCode::Memory;
		return true;
	case BarrierKind::Full:
		code = Node::BarrierCode::Full;
		return true;
	default:
		return false;
	}
}

void AppendIndentedCode(std::string &code, const std::vector<std::string> &lines) {
	for (const auto &line : lines) {
		if (line.empty()) {
			continue;
		}

		code += "    ";
		code += line;
		if (line.back() != '\n') {
			code += "\n";
		}
	}
}

std::string TrimHeaderStatement(std::string code) {
	while (!code.empty() && (code.back() == '\n' || code.back() == '\r' || code.back() == '\t' || code.back() == ' ')) {
		code.pop_back();
	}
	if (!code.empty() && code.back() == ';') {
		code.pop_back();
	}
	while (!code.empty() && (code.back() == '\t' || code.back() == ' ')) {
		code.pop_back();
	}
	return code;
}

bool IsLocalValue(const ValueRecord &value, const std::string &name) {
	return value.kind == ValueRecord::Kind::LocalVar && value.localName == name;
}

class ModuleLowerer {
public:
	explicit ModuleLowerer(const Module &module, GPU::AD::GradientTape* tape = nullptr) : _module(module), _gradientTape(tape) {
	}

	[[nodiscard]] std::unique_ptr<Kernel::KernelBuildContext> Build() {
		if (_module.functions.size() != 1 || _module.functions.front().stage != ShaderStage::Compute) {
			return nullptr;
		}

		const auto &function = _module.functions.front();
		const auto dimension = function.dimension > 0 ? function.dimension : (function.workSizeZ > 1 ? 3 : function.workSizeY > 1 ? 2 : 1);
		auto	   context	= std::make_unique<Kernel::KernelBuildContext>(dimension);
		context->WorkSizeX = function.workSizeX;
		context->WorkSizeY = function.workSizeY;
		context->WorkSizeZ = function.workSizeZ;
		if (!RegisterResources(*context)) {
			return nullptr;
		}

			auto &builder = Builder::Builder::Get();
			Builder::Builder::ScopedBind bind(builder, *context);
			Builder::Builder::ScopedGradientTape tapeScope(builder, _gradientTape);
			if (!RegisterCallableFunctions(*context)) {
				return nullptr;
			}
		for (const auto &statement : function.statements) {
			if (!LowerStatement(statement)) {
				return nullptr;
			}
		}

		return context;
	}

private:
	[[nodiscard]] bool RegisterCallableFunctions(Kernel::KernelBuildContext &context) {
		for (const auto &callable : _module.callables) {
			const auto declaration = CallableDeclaration(callable);
			if (declaration.empty()) {
				return false;
			}

			context.AddCallableDeclaration(declaration);
			if (!EmitCallableBody(context, callable.id)) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] bool RegisterResources(Kernel::KernelBuildContext &context) {
		RegisterStructDefinitions(context);
		for (const auto &resource : _module.resources) {
			const auto type = ToGlslType(resource.elementType);
			if (type.empty()) {
				return false;
			}

			if (resource.kind == ResourceKind::Buffer) {
				context.RegisterBuffer(resource.binding, type, resource.name, ToBackendBufferMode(resource.access));
				continue;
			}

			if (resource.kind == ResourceKind::Texture) {
				if (resource.width == 0 || resource.height == 0 || resource.depth == 0) {
					return false;
				}

				if (resource.textureDimension == 3) {
					context.RegisterTexture3D(resource.binding, resource.textureFormat, resource.name, resource.width,
											  resource.height, resource.depth, resource.sampled);
				} else {
					context.RegisterTexture(resource.binding, resource.textureFormat, resource.name, resource.width,
											resource.height, resource.sampled);
				}
				continue;
			}

			if (resource.kind == ResourceKind::PushConstant) {
				if (resource.size == 0 || resource.alignment == 0) {
					return false;
				}

				auto packFunc = [size = resource.size](void *dst, void *ptr) {
					if (dst == nullptr || ptr == nullptr) {
						return;
					}
					std::memcpy(dst, ptr, size);
				};
				_uniformNames[resource.id] =
					context.RegisterUniform(type, resource.data, resource.size, resource.alignment, nullptr, packFunc);
				continue;
			}

			return false;
		}

		return true;
	}

	[[nodiscard]] std::string CallableDeclaration(const CallableFunction &callable) const {
		if (callable.id >= _module.callables.size() || callable.name.empty()) {
			return {};
		}

		const auto returnType = ToGlslType(callable.returnType);
		if (returnType.empty()) {
			return {};
		}

		std::string declaration = returnType + " " + callable.name + "(";
		for (size_t i = 0; i < callable.parameters.size(); ++i) {
			const auto parameterType = ToGlslType(callable.parameters[i].second);
			if (parameterType.empty() || callable.parameters[i].first.empty()) {
				return {};
			}

			if (i > 0) {
				declaration += ", ";
			}
			declaration += parameterType + " " + callable.parameters[i].first;
		}
		declaration += ")";
		return declaration;
	}

	[[nodiscard]] bool EmitCallableBody(Kernel::KernelBuildContext &context, FunctionId callableId) {
		if (callableId >= _module.callables.size()) {
			return false;
		}

		bool ok = true;
		const auto previousBlocks = _activeBlocks;
		_activeBlocks = &_module.callables[callableId].blocks;
		context.PushCallableBody();
		for (const auto &statement : _module.callables[callableId].statements) {
			if (!LowerStatement(statement)) {
				ok = false;
				break;
			}
		}
		context.PopCallableBody();
		_activeBlocks = previousBlocks;
		return ok;
	}

	void RegisterStructDefinitions(Kernel::KernelBuildContext &context) {
		for (const auto &resource : _module.resources) {
			RegisterStructDefinition(context, resource.elementType);
		}
		for (const auto &value : _module.values) {
			RegisterStructDefinition(context, value.type);
		}
		for (const auto &function : _module.functions) {
			for (const auto &statement : function.statements) {
				RegisterStructDefinition(context, statement.localType);
				RegisterStructDefinition(context, statement.sharedType);
			}
			for (const auto &block : function.blocks) {
				for (const auto &statement : block.statements) {
					RegisterStructDefinition(context, statement.localType);
					RegisterStructDefinition(context, statement.sharedType);
				}
			}
		}
		for (const auto &callable : _module.callables) {
			RegisterStructDefinition(context, callable.returnType);
			for (const auto &parameter : callable.parameters) {
				RegisterStructDefinition(context, parameter.second);
			}
			for (const auto &statement : callable.statements) {
				RegisterStructDefinition(context, statement.localType);
				RegisterStructDefinition(context, statement.sharedType);
			}
			for (const auto &block : callable.blocks) {
				for (const auto &statement : block.statements) {
					RegisterStructDefinition(context, statement.localType);
					RegisterStructDefinition(context, statement.sharedType);
				}
			}
		}
	}

	static void RegisterStructDefinition(Kernel::KernelBuildContext &context, const Type &type) {
		if (type.kind == Type::Kind::Struct && !type.typeName.empty() && !type.definition.empty()) {
			for (const auto &dependency : type.dependencyDefinitions) {
				if (!dependency.first.empty() && !dependency.second.empty()) {
					context.AddStructDefinition(dependency.first, dependency.second);
				}
			}

			context.AddStructDefinition(type.typeName, type.definition);
		}
	}

	[[nodiscard]] bool LowerStatement(const Statement &statement) {
		switch (statement.kind) {
		case Statement::Kind::LocalDeclaration:
			return LowerLocalDeclaration(statement);
		case Statement::Kind::Store: {
			if (statement.target < _module.values.size() &&
				_module.values[statement.target].kind == ValueRecord::Kind::TextureElement) {
				return LowerTextureStore(_module.values[statement.target], statement.value);
			}

			auto target = BuildNode(statement.target);
			auto value	= BuildNode(statement.value);
			if (target == nullptr || value == nullptr) {
				return false;
			}

			const Node::StoreNode store(std::move(target), std::move(value));
			Builder::Builder::Get().Build(store, true);
			return true;
		}
		case Statement::Kind::If:
			return LowerIfStatement(statement);
		case Statement::Kind::For:
			return LowerForStatement(statement);
		case Statement::Kind::While:
			return LowerWhileStatement(statement);
		case Statement::Kind::DoWhile:
			return LowerDoWhileStatement(statement);
		case Statement::Kind::Break: {
			const Node::BreakNode breakNode;
			Builder::Builder::Get().Build(breakNode, true);
			return true;
		}
		case Statement::Kind::Continue: {
			const Node::ContinueNode continueNode;
			Builder::Builder::Get().Build(continueNode, true);
			return true;
		}
		case Statement::Kind::Return: {
			if (statement.value != InvalidValueId) {
				auto value = BuildNode(statement.value);
				if (value == nullptr) return false;
				const Node::ReturnNode returnNode(std::move(value));
				Builder::Builder::Get().Build(returnNode, true);
			} else {
				const Node::ReturnNode returnNode;
				Builder::Builder::Get().Build(returnNode, true);
			}
			return true;
		}
		case Statement::Kind::Expression: {
			auto value = BuildNode(statement.value);
			if (value == nullptr) {
				return false;
			}

			Builder::Builder::Get().Build(*value, true);
			return true;
		}
		case Statement::Kind::RawGLSL:
			// Compatibility escape hatch for legacy callers. Normal section-7
			// constructs lower through typed module values/statements and EasyGPU
			// nodes before reaching backend GLSL emission.
			Builder::Builder::Get().ContextChecked()->PushTranslatedCode(statement.rawGlsl);
			return true;
		case Statement::Kind::Barrier:
			return LowerBarrierStatement(statement);
		case Statement::Kind::SharedMemoryDecl:
			return LowerSharedMemoryDeclaration(statement);
		default:
			return false;
		}
	}

	[[nodiscard]] bool LowerTextureStore(const ValueRecord &target, ValueId source) {
		if (target.resource >= _module.resources.size()) {
			return false;
		}

		const auto &resource = _module.resources[target.resource];
		if (resource.kind != ResourceKind::Texture || resource.sampled) {
			return false;
		}

		auto x = BuildNode(target.index);
		auto y = BuildNode(target.y);
		auto z = target.right == InvalidValueId ? nullptr : BuildNode(target.right);
		auto value = BuildNode(source);
		if (x == nullptr || y == nullptr || (target.right != InvalidValueId && z == nullptr) || value == nullptr) {
			return false;
		}

		const auto store = z == nullptr
			? Node::TextureStoreNode(resource.name, std::move(x), std::move(y), std::move(value))
			: Node::TextureStoreNode(resource.name, std::move(x), std::move(y), std::move(z), std::move(value));
		Builder::Builder::Get().Build(store, true);
		return true;
	}

	[[nodiscard]] bool LowerBarrierStatement(const Statement &statement) {
		Node::BarrierCode code{};
		switch (statement.barrierKind) {
		case BarrierKind::Workgroup:
			code = Node::BarrierCode::Workgroup;
			break;
		case BarrierKind::Memory:
			code = Node::BarrierCode::Memory;
			break;
		case BarrierKind::Full:
			code = Node::BarrierCode::Full;
			break;
		default:
			return false;
		}

		const Node::BarrierNode barrierNode(code);
		Builder::Builder::Get().Build(barrierNode, true);
		return true;
	}

	[[nodiscard]] bool LowerSharedMemoryDeclaration(const Statement &statement) {
		const auto typeStr = ToGlslType(statement.sharedType);
		if (typeStr.empty() || statement.sharedName.empty() || statement.sharedCount == 0) {
			return false;
		}

		const Node::SharedMemoryNode sharedMemory(statement.sharedName, typeStr, static_cast<int>(statement.sharedCount));
		Builder::Builder::Get().Build(sharedMemory, true);
		return true;
	}

	[[nodiscard]] bool LowerLocalDeclaration(const Statement &statement) {
		const auto type = ToGlslType(statement.localType);
		if (type.empty() || statement.localName.empty()) {
			return false;
		}

		std::unique_ptr<Node::Node> initializer;
		if (statement.initializer != InvalidValueId) {
			initializer = BuildNode(statement.initializer);
			if (initializer == nullptr) {
				return false;
			}
		}

		auto declaration = initializer == nullptr
			? Node::LocalVariableNode(statement.localName, type)
			: Node::LocalVariableNode(statement.localName, type, std::move(initializer));
		Builder::Builder::Get().Build(declaration, true);
		return true;
	}

	[[nodiscard]] bool LowerStatementToNodes(const Statement &statement, std::vector<std::unique_ptr<Node::Node>> &nodes) {
		switch (statement.kind) {
		case Statement::Kind::LocalDeclaration: {
			const auto type = ToGlslType(statement.localType);
			if (type.empty() || statement.localName.empty()) {
				return false;
			}

			std::unique_ptr<Node::Node> initializer;
			if (statement.initializer != InvalidValueId) {
				initializer = BuildNode(statement.initializer);
				if (initializer == nullptr) {
					return false;
				}
			}

			nodes.push_back(initializer == nullptr
				? std::make_unique<Node::LocalVariableNode>(statement.localName, type)
				: std::make_unique<Node::LocalVariableNode>(statement.localName, type, std::move(initializer)));
			return true;
		}
		case Statement::Kind::Store: {
			if (statement.target >= _module.values.size()) {
				return false;
			}

			if (_module.values[statement.target].kind == ValueRecord::Kind::TextureElement) {
				const auto &target = _module.values[statement.target];
				if (target.resource >= _module.resources.size()) {
					return false;
				}

				const auto &resource = _module.resources[target.resource];
				if (resource.kind != ResourceKind::Texture || resource.sampled) {
					return false;
				}

				auto x = BuildNode(target.index);
				auto y = BuildNode(target.y);
				auto z = target.right == InvalidValueId ? nullptr : BuildNode(target.right);
				auto value = BuildNode(statement.value);
				if (x == nullptr || y == nullptr || (target.right != InvalidValueId && z == nullptr) || value == nullptr) {
					return false;
				}

				if (z == nullptr) {
					nodes.push_back(std::make_unique<Node::TextureStoreNode>(
						resource.name, std::move(x), std::move(y), std::move(value)));
				} else {
					nodes.push_back(std::make_unique<Node::TextureStoreNode>(
						resource.name, std::move(x), std::move(y), std::move(z), std::move(value)));
				}
				return true;
			}

			auto target = BuildNode(statement.target);
			auto value = BuildNode(statement.value);
			if (target == nullptr || value == nullptr) {
				return false;
			}

			nodes.push_back(std::make_unique<Node::StoreNode>(std::move(target), std::move(value)));
			return true;
		}
		case Statement::Kind::If:
			return LowerIfStatementToNodes(statement, nodes);
		case Statement::Kind::For:
			return LowerForStatementToNodes(statement, nodes);
		case Statement::Kind::While:
			return LowerWhileStatementToNodes(statement, nodes);
		case Statement::Kind::DoWhile:
			return LowerDoWhileStatementToNodes(statement, nodes);
		case Statement::Kind::Break:
			nodes.push_back(std::make_unique<Node::BreakNode>());
			return true;
		case Statement::Kind::Continue:
			nodes.push_back(std::make_unique<Node::ContinueNode>());
			return true;
		case Statement::Kind::Return: {
			if (statement.value == InvalidValueId) {
				nodes.push_back(std::make_unique<Node::ReturnNode>());
				return true;
			}

			auto value = BuildNode(statement.value);
			if (value == nullptr) {
				return false;
			}

			nodes.push_back(std::make_unique<Node::ReturnNode>(std::move(value)));
			return true;
		}
		case Statement::Kind::Expression: {
			auto value = BuildNode(statement.value);
			if (value == nullptr) {
				return false;
			}

			nodes.push_back(std::move(value));
			return true;
		}
		case Statement::Kind::Barrier: {
			Node::BarrierCode code{};
			if (!TryMapBarrierCode(statement.barrierKind, code)) {
				return false;
			}

			nodes.push_back(std::make_unique<Node::BarrierNode>(code));
			return true;
		}
		case Statement::Kind::SharedMemoryDecl: {
			const auto typeStr = ToGlslType(statement.sharedType);
			if (typeStr.empty() || statement.sharedName.empty() || statement.sharedCount == 0) {
				return false;
			}

			nodes.push_back(std::make_unique<Node::SharedMemoryNode>(
				statement.sharedName, typeStr, static_cast<int>(statement.sharedCount)));
			return true;
		}
		case Statement::Kind::RawGLSL:
			nodes.push_back(std::make_unique<Node::RawCodeNode>(statement.rawGlsl));
			return true;
		default:
			return false;
		}
	}

	[[nodiscard]] std::optional<std::vector<std::unique_ptr<Node::Node>>> BuildStatementNodes(BlockId block) {
		std::vector<std::unique_ptr<Node::Node>> nodes;
		const auto &blocks = ActiveBlocks();
		if (block >= blocks.size()) {
			return std::move(nodes);
		}

		for (const auto &statement : blocks[block].statements) {
			if (!LowerStatementToNodes(statement, nodes)) {
				return std::nullopt;
			}
		}

		return std::move(nodes);
	}

	[[nodiscard]] bool RecordBlockOnTape(BlockId blockId) {
		auto *tape = Builder::Builder::Get().GetGradientTape();
		if (tape == nullptr) {
			return true;
		}

		const auto *block = FindActiveBlock(blockId);
		if (block == nullptr) {
			return true;
		}

		for (const auto &statement : block->statements) {
			if (!RecordStatementOnTape(statement, *tape)) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] bool RecordStatementOnTape(const Statement &statement, GPU::AD::GradientTape &tape) {
		auto &builder = Builder::Builder::Get();
		switch (statement.kind) {
		case Statement::Kind::If: {
			auto condition = BuildNode(statement.condition);
			if (condition == nullptr) return false;
			const auto condStr = builder.BuildNode(*condition);
			if (condStr.empty()) return false;

			tape.BeginIfBranch(condStr);
			if (!RecordBlockOnTape(statement.thenBlock)) return false;
			if (statement.elseBlock != InvalidBlockId) {
				tape.BeginElseBranch();
				if (!RecordBlockOnTape(statement.elseBlock)) return false;
			}
			tape.EndIfChain();
			return true;
		}
		case Statement::Kind::For: {
			auto tapeInfo = TryExtractForTapeInfo(statement);
			if (!tapeInfo.has_value()) return false;
			tape.BeginForLoop(tapeInfo->varName, tapeInfo->start, tapeInfo->end, tapeInfo->step);
			if (!RecordBlockOnTape(statement.bodyBlock)) return false;
			tape.EndForLoop();
			return true;
		}
		case Statement::Kind::While:
		case Statement::Kind::DoWhile:
		case Statement::Kind::Break:
		case Statement::Kind::Continue:
			return true;
		default: {
			std::vector<std::unique_ptr<Node::Node>> nodes;
			if (!LowerStatementToNodes(statement, nodes)) {
				return false;
			}
			for (const auto &node : nodes) {
				if (node != nullptr) {
					tape.Record(*node, true);
				}
			}
			return true;
		}
		}
	}

	[[nodiscard]] const std::vector<Block> &ActiveBlocks() const {
		return _activeBlocks == nullptr ? _module.functions.front().blocks : *_activeBlocks;
	}

	[[nodiscard]] const Block *FindActiveBlock(BlockId blockId) const {
		if (blockId == InvalidBlockId) {
			return nullptr;
		}

		const auto &blocks = ActiveBlocks();
		if (blockId >= blocks.size()) {
			return nullptr;
		}

		return &blocks[blockId];
	}

	[[nodiscard]] std::optional<std::vector<std::string>> LowerBlockToCollectedCode(BlockId blockId) {
		const auto *block = FindActiveBlock(blockId);
		if (block == nullptr) {
			return std::vector<std::string>{};
		}

		auto &builder = Builder::Builder::Get();
		auto *parent = builder.Context();
		if (parent == nullptr) {
			return std::nullopt;
		}

		Flow::CodeCollectContext collectContext;
		collectContext.SetParentContext(parent);
		{
			Builder::Builder::ScopedBind bind(builder, collectContext);
			for (const auto &statement : block->statements) {
				if (!LowerStatement(statement)) {
					return std::nullopt;
				}
			}
		}

		return collectContext.ReleaseCollectedCode();
	}

	[[nodiscard]] std::optional<std::string> BuildForHeader(BlockId blockId) {
		const auto *block = FindActiveBlock(blockId);
		if (block == nullptr) {
			return std::string{};
		}

		auto &builder = Builder::Builder::Get();
		std::string header;
		for (const auto &statement : block->statements) {
			std::vector<std::unique_ptr<Node::Node>> nodes;
			if (!LowerStatementToNodes(statement, nodes)) {
				return std::nullopt;
			}

			for (const auto &node : nodes) {
				if (!node) {
					continue;
				}

				auto part = TrimHeaderStatement(builder.BuildNode(*node));
				if (part.empty()) {
					continue;
				}
				if (!header.empty()) {
					header += ", ";
				}
				header += part;
			}
		}

		return header;
	}

	struct ForTapeInfo {
		std::string varName;
		std::string start;
		std::string end;
		std::string step;
	};

	[[nodiscard]] std::optional<ForTapeInfo> TryExtractForTapeInfo(const Statement &statement) {
		if (statement.condition >= _module.values.size()) {
			return std::nullopt;
		}

		const auto &condition = _module.values[statement.condition];
		if (condition.kind != ValueRecord::Kind::Compare || condition.compareOp != CompareOp::Less ||
			condition.left >= _module.values.size() || condition.right >= _module.values.size()) {
			return std::nullopt;
		}

		const auto &left = _module.values[condition.left];
		if (left.kind != ValueRecord::Kind::LocalVar || left.localName.empty()) {
			return std::nullopt;
		}

		ForTapeInfo info;
		info.varName = left.localName;

		auto start = TryExtractForStart(statement.initBlock, info.varName);
		auto step = TryExtractForStep(statement.stepBlock, info.varName);
		auto endNode = BuildNode(condition.right);
		if (!start.has_value() || !step.has_value() || endNode == nullptr) {
			return std::nullopt;
		}

		auto &builder = Builder::Builder::Get();
		info.start = *start;
		info.end = builder.BuildNode(*endNode);
		info.step = *step;
		if (info.start.empty() || info.end.empty() || info.step.empty()) {
			return std::nullopt;
		}

		return info;
	}

	[[nodiscard]] std::optional<std::string> TryExtractForStart(BlockId blockId, const std::string &varName) {
		const auto *block = FindActiveBlock(blockId);
		if (block == nullptr) {
			return std::nullopt;
		}

		for (const auto &statement : block->statements) {
			ValueId initializer = InvalidValueId;
			if (statement.kind == Statement::Kind::LocalDeclaration && statement.localName == varName) {
				initializer = statement.initializer;
			} else if (statement.kind == Statement::Kind::Store && statement.target < _module.values.size() &&
					   IsLocalValue(_module.values[statement.target], varName)) {
				initializer = statement.value;
			}

			if (initializer == InvalidValueId) {
				continue;
			}

			auto node = BuildNode(initializer);
			if (node == nullptr) {
				return std::nullopt;
			}
			auto text = Builder::Builder::Get().BuildNode(*node);
			return text.empty() ? std::nullopt : std::optional<std::string>(std::move(text));
		}

		return std::nullopt;
	}

	[[nodiscard]] std::optional<std::string> TryExtractForStep(BlockId blockId, const std::string &varName) {
		const auto *block = FindActiveBlock(blockId);
		if (block == nullptr) {
			return std::nullopt;
		}

		for (const auto &statement : block->statements) {
			if (statement.kind != Statement::Kind::Store || statement.target >= _module.values.size() ||
				!IsLocalValue(_module.values[statement.target], varName) || statement.value >= _module.values.size()) {
				continue;
			}

			const auto &value = _module.values[statement.value];
			if (value.kind != ValueRecord::Kind::Binary || value.binaryOp != BinaryOp::Add ||
				value.left >= _module.values.size() || value.right >= _module.values.size()) {
				continue;
			}

			ValueId stepValue = InvalidValueId;
			if (IsLocalValue(_module.values[value.left], varName)) {
				stepValue = value.right;
			} else if (IsLocalValue(_module.values[value.right], varName)) {
				stepValue = value.left;
			}

			if (stepValue == InvalidValueId) {
				continue;
			}

			auto node = BuildNode(stepValue);
			if (node == nullptr) {
				return std::nullopt;
			}
			auto text = Builder::Builder::Get().BuildNode(*node);
			return text.empty() ? std::nullopt : std::optional<std::string>(std::move(text));
		}

		return std::nullopt;
	}

	[[nodiscard]] bool LowerIfStatement(const Statement &statement) {
		auto &builder = Builder::Builder::Get();

		if (auto *tape = builder.GetGradientTape()) {
			if (!RecordStatementOnTape(statement, *tape)) {
				return false;
			}
		}

		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		auto thenNodes = BuildStatementNodes(statement.thenBlock);
		if (!thenNodes.has_value()) return false;

		std::vector<std::unique_ptr<Node::Node>> elseNodes;
		if (statement.elseBlock != InvalidBlockId) {
			auto builtElse = BuildStatementNodes(statement.elseBlock);
			if (!builtElse.has_value()) return false;
			elseNodes = std::move(*builtElse);
		}

		std::vector<std::pair<std::unique_ptr<Node::Node>, std::vector<std::unique_ptr<Node::Node>>>> elifs;
		const Node::IfNode ifNode(*thenNodes, condition, elifs, elseNodes);
		Builder::Builder::ScopedGradientTape suppressTape(builder, nullptr);
		Builder::Builder::Get().Build(ifNode, true);
		return true;
	}

	[[nodiscard]] bool LowerIfStatementToNodes(const Statement &statement,
											   std::vector<std::unique_ptr<Node::Node>> &nodes) {
		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		std::vector<std::unique_ptr<Node::Node>> thenNodes;
		const auto &blocks = ActiveBlocks();
		if (statement.thenBlock < blocks.size()) {
			auto builtThen = BuildStatementNodes(statement.thenBlock);
			if (!builtThen.has_value()) return false;
			thenNodes = std::move(*builtThen);
		}

		std::vector<std::pair<std::unique_ptr<Node::Node>, std::vector<std::unique_ptr<Node::Node>>>> elifs;
		std::vector<std::unique_ptr<Node::Node>> elseNodes;
		if (statement.elseBlock < blocks.size()) {
			auto builtElse = BuildStatementNodes(statement.elseBlock);
			if (!builtElse.has_value()) return false;
			elseNodes = std::move(*builtElse);
		}

		nodes.push_back(std::make_unique<Node::IfNode>(thenNodes, condition, elifs, elseNodes));
		return true;
	}

	[[nodiscard]] bool LowerForStatement(const Statement &statement) {
		auto &builder = Builder::Builder::Get();
		auto cond = BuildNode(statement.condition);
		if (cond == nullptr) return false;

		auto initHeader = BuildForHeader(statement.initBlock);
		auto stepHeader = BuildForHeader(statement.stepBlock);
		if (!initHeader.has_value() || !stepHeader.has_value()) return false;

		if (auto *tape = builder.GetGradientTape()) {
			if (!RecordStatementOnTape(statement, *tape)) {
				return false;
			}
		}

		std::vector<std::unique_ptr<Node::Node>> initNodes;
		if (statement.initBlock < ActiveBlocks().size()) {
			auto builtInit = BuildStatementNodes(statement.initBlock);
			if (!builtInit.has_value()) return false;
			initNodes = std::move(*builtInit);
		}

		std::vector<std::unique_ptr<Node::Node>> stepNodes;
		if (statement.stepBlock < ActiveBlocks().size()) {
			auto builtStep = BuildStatementNodes(statement.stepBlock);
			if (!builtStep.has_value()) return false;
			stepNodes = std::move(*builtStep);
		}

		auto bodyNodes = BuildStatementNodes(statement.bodyBlock);
		if (!bodyNodes.has_value()) return false;

		const Node::ForNode forNode(initNodes, cond, stepNodes, *bodyNodes);
		Builder::Builder::ScopedGradientTape suppressTape(builder, nullptr);
		Builder::Builder::Get().Build(forNode, true);
		return true;
	}

	[[nodiscard]] bool LowerForStatementToNodes(const Statement &statement,
												std::vector<std::unique_ptr<Node::Node>> &nodes) {
		auto cond = BuildNode(statement.condition);
		if (cond == nullptr) return false;

		std::vector<std::unique_ptr<Node::Node>> initNodes;
		if (statement.initBlock < ActiveBlocks().size()) {
			auto builtInit = BuildStatementNodes(statement.initBlock);
			if (!builtInit.has_value()) return false;
			initNodes = std::move(*builtInit);
		}

		std::vector<std::unique_ptr<Node::Node>> stepNodes;
		if (statement.stepBlock < ActiveBlocks().size()) {
			auto builtStep = BuildStatementNodes(statement.stepBlock);
			if (!builtStep.has_value()) return false;
			stepNodes = std::move(*builtStep);
		}

		auto bodyNodes = BuildStatementNodes(statement.bodyBlock);
		if (!bodyNodes.has_value()) return false;

		nodes.push_back(std::make_unique<Node::ForNode>(initNodes, cond, stepNodes, *bodyNodes));
		return true;
	}

	[[nodiscard]] bool LowerWhileStatement(const Statement &statement) {
		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		auto builtBody = BuildStatementNodes(statement.bodyBlock);
		if (!builtBody.has_value()) return false;
		auto bodyNodes = std::move(*builtBody);

		const Node::WhileNode whileNode(condition, bodyNodes);
		Builder::Builder::Get().Build(whileNode, true);
		return true;
	}

	[[nodiscard]] bool LowerWhileStatementToNodes(const Statement &statement,
												  std::vector<std::unique_ptr<Node::Node>> &nodes) {
		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		auto bodyNodes = BuildStatementNodes(statement.bodyBlock);
		if (!bodyNodes.has_value()) return false;

		nodes.push_back(std::make_unique<Node::WhileNode>(condition, *bodyNodes));
		return true;
	}

	[[nodiscard]] bool LowerDoWhileStatement(const Statement &statement) {
		auto builtBody = BuildStatementNodes(statement.bodyBlock);
		if (!builtBody.has_value()) return false;
		auto bodyNodes = std::move(*builtBody);

		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		const Node::DoWhileNode doWhileNode(bodyNodes, condition);
		Builder::Builder::Get().Build(doWhileNode, true);
		return true;
	}

	[[nodiscard]] bool LowerDoWhileStatementToNodes(const Statement &statement,
													std::vector<std::unique_ptr<Node::Node>> &nodes) {
		auto bodyNodes = BuildStatementNodes(statement.bodyBlock);
		if (!bodyNodes.has_value()) return false;

		auto condition = BuildNode(statement.condition);
		if (condition == nullptr) return false;

		nodes.push_back(std::make_unique<Node::DoWhileNode>(*bodyNodes, condition));
		return true;
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildCompare(const ValueRecord &value) {
		if (value.left >= _module.values.size() || value.right >= _module.values.size()) return nullptr;
		auto left  = BuildNode(value.left);
		auto right = BuildNode(value.right);
		if (left == nullptr || right == nullptr) return nullptr;

		Node::OperationCode code;
		switch (value.compareOp) {
		case CompareOp::Less:		code = Node::OperationCode::Less; break;
		case CompareOp::LessEqual:	code = Node::OperationCode::LessEqual; break;
		case CompareOp::Greater:	code = Node::OperationCode::Greater; break;
		case CompareOp::GreaterEqual: code = Node::OperationCode::GreaterEqual; break;
		case CompareOp::Equal:		code = Node::OperationCode::Equal; break;
		case CompareOp::NotEqual:	code = Node::OperationCode::NotEqual; break;
		default: return nullptr;
		}
		return std::make_unique<Node::OperationNode>(code, std::move(left), std::move(right));
	}


	[[nodiscard]] std::unique_ptr<Node::Node> BuildNode(ValueId id) {
		if (id >= _module.values.size()) {
			return nullptr;
		}

		const auto &value = _module.values[id];
		switch (value.kind) {
		case ValueRecord::Kind::ThreadIndexX:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_GlobalInvocationID.x)");
		case ValueRecord::Kind::ThreadIndexY:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_GlobalInvocationID.y)");
		case ValueRecord::Kind::ThreadIndexZ:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_GlobalInvocationID.z)");
		case ValueRecord::Kind::LocalIndexX:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_LocalInvocationID.x)");
		case ValueRecord::Kind::LocalIndexY:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_LocalInvocationID.y)");
		case ValueRecord::Kind::LocalIndexZ:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_LocalInvocationID.z)");
		case ValueRecord::Kind::GroupIdX:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupID.x)");
		case ValueRecord::Kind::GroupIdY:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupID.y)");
		case ValueRecord::Kind::GroupIdZ:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupID.z)");
		case ValueRecord::Kind::DispatchSizeX:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_NumWorkGroups.x * gl_WorkGroupSize.x)");
		case ValueRecord::Kind::DispatchSizeY:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_NumWorkGroups.y * gl_WorkGroupSize.y)");
		case ValueRecord::Kind::DispatchSizeZ:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_NumWorkGroups.z * gl_WorkGroupSize.z)");
		case ValueRecord::Kind::GroupSizeX:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupSize.x)");
		case ValueRecord::Kind::GroupSizeY:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupSize.y)");
		case ValueRecord::Kind::GroupSizeZ:
			return std::make_unique<Node::LoadLocalVariableNode>("int(gl_WorkGroupSize.z)");
		case ValueRecord::Kind::ResourceElement:
			return BuildResourceElement(value);
		case ValueRecord::Kind::TextureElement:
			return BuildTextureElement(value);
		case ValueRecord::Kind::PushConstant:
			return BuildPushConstant(value);
		case ValueRecord::Kind::Literal:
			return std::make_unique<Node::LoadUniformNode>(value.literal);
		case ValueRecord::Kind::LocalVar:
			return std::make_unique<Node::LoadLocalVariableNode>(value.localName);
		case ValueRecord::Kind::Ternary: {
			auto cond = BuildNode(value.left);
			auto tv = BuildNode(value.right);
			auto fv = BuildNode(value.arguments[0]);
			if (cond == nullptr || tv == nullptr || fv == nullptr) return nullptr;
			return std::make_unique<Node::TernaryNode>(std::move(cond), std::move(tv), std::move(fv));
		}
		case ValueRecord::Kind::Binary:
			return BuildBinary(value);
		case ValueRecord::Kind::Unary:
			return BuildUnary(value);
		case ValueRecord::Kind::Compare:
			return BuildCompare(value);
		case ValueRecord::Kind::Intrinsic:
			return BuildIntrinsic(value);
		case ValueRecord::Kind::TextureSample:
			return BuildTextureSample(value, false);
		case ValueRecord::Kind::TextureSampleLevel:
			return BuildTextureSample(value, true);
		case ValueRecord::Kind::Call:
			return BuildCall(value);
		case ValueRecord::Kind::Swizzle:
			return BuildSwizzle(value);
		case ValueRecord::Kind::IndexAccess:
			return BuildIndexAccess(value);
		case ValueRecord::Kind::MemberAccess:
			return BuildMemberAccess(value);
		case ValueRecord::Kind::SharedMemoryElement:
			return BuildSharedMemoryElement(value);
		case ValueRecord::Kind::Atomic:
			return BuildAtomic(value);
		default:
			return nullptr;
		}
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildResourceElement(const ValueRecord &value) {
		if (value.resource >= _module.resources.size()) {
			return nullptr;
		}

		auto target = std::make_unique<Node::LoadLocalVariableNode>(_module.resources[value.resource].name);
		auto index	= BuildNode(value.index);
		if (index == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::ArrayAccessNode>(std::move(target), std::move(index));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildTextureElement(const ValueRecord &value) {
		if (value.resource >= _module.resources.size()) {
			return nullptr;
		}

		const auto &resource = _module.resources[value.resource];
		if (resource.kind != ResourceKind::Texture) {
			return nullptr;
		}

		auto x = BuildNode(value.index);
		auto y = BuildNode(value.y);
		auto z = value.right == InvalidValueId ? nullptr : BuildNode(value.right);
		if (x == nullptr || y == nullptr || (value.right != InvalidValueId && z == nullptr)) {
			return nullptr;
		}

		if (z == nullptr) {
			return std::make_unique<Node::TextureLoadNode>(resource.name, std::move(x), std::move(y));
		}

		return std::make_unique<Node::TextureLoadNode>(resource.name, std::move(x), std::move(y), std::move(z));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildPushConstant(const ValueRecord &value) {
		const auto uniformName = _uniformNames.find(value.resource);
		if (uniformName == _uniformNames.end()) {
			return nullptr;
		}

		return std::make_unique<Node::LoadUniformNode>(uniformName->second);
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildBinary(const ValueRecord &value) {
		auto left  = BuildNode(value.left);
		auto right = BuildNode(value.right);
		if (left == nullptr || right == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::OperationNode>(ToNodeOperation(value.binaryOp), std::move(left), std::move(right));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildUnary(const ValueRecord &value) {
		auto operand = BuildNode(value.left);
		if (operand == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::OperationNode>(ToNodeOperation(value.unaryOp), std::move(operand));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildIntrinsic(const ValueRecord &value) {
		std::vector<std::unique_ptr<Node::Node>> arguments;
		arguments.reserve(value.arguments.size());
		for (const auto argument : value.arguments) {
			auto node = BuildNode(argument);
			if (node == nullptr) {
				return nullptr;
			}

			arguments.push_back(std::move(node));
		}

		return std::make_unique<Node::IntrinsicCallNode>(value.intrinsic, std::move(arguments));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildTextureSample(const ValueRecord &value, bool explicitLevel) {
		if (value.resource >= _module.resources.size()) {
			return nullptr;
		}

		const auto &resource = _module.resources[value.resource];
		if (resource.kind != ResourceKind::Texture || !resource.sampled || value.arguments.empty()) {
			return nullptr;
		}

		auto uv = BuildNode(value.arguments[0]);
		if (uv == nullptr) {
			return nullptr;
		}

		if (!explicitLevel) {
			return std::make_unique<Node::TextureSampleNode>(resource.name, std::move(uv));
		}

		if (value.arguments.size() < 2) {
			return nullptr;
		}

		auto lod = BuildNode(value.arguments[1]);
		if (lod == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::TextureSampleNode>(resource.name, std::move(uv), std::move(lod));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildAtomic(const ValueRecord &value) {
		auto target = BuildNode(value.left);
		if (target == nullptr || value.arguments.empty()) {
			return nullptr;
		}

		auto operand = BuildNode(value.arguments[0]);
		if (operand == nullptr) {
			return nullptr;
		}

		if (value.atomicOp == AtomicOp::CompareExchange) {
			if (value.arguments.size() != 2) {
				return nullptr;
			}

			auto compare = std::move(operand);
			auto replacement = BuildNode(value.arguments[1]);
			if (replacement == nullptr) {
				return nullptr;
			}

			return std::make_unique<Node::AtomicOpNode>(
				std::move(target), std::move(compare), std::move(replacement));
		}

		Node::AtomicOpCode code{};
		switch (value.atomicOp) {
		case AtomicOp::Add:
			code = Node::AtomicOpCode::Add;
			break;
		case AtomicOp::Sub:
			code = Node::AtomicOpCode::Sub;
			break;
		case AtomicOp::Min:
			code = Node::AtomicOpCode::Min;
			break;
		case AtomicOp::Max:
			code = Node::AtomicOpCode::Max;
			break;
		case AtomicOp::And:
			code = Node::AtomicOpCode::And;
			break;
		case AtomicOp::Or:
			code = Node::AtomicOpCode::Or;
			break;
		case AtomicOp::Xor:
			code = Node::AtomicOpCode::Xor;
			break;
		case AtomicOp::Exchange:
			code = Node::AtomicOpCode::Exchange;
			break;
		default:
			return nullptr;
		}

		if (value.arguments.size() != 1) {
			return nullptr;
		}

		return std::make_unique<Node::AtomicOpNode>(code, std::move(target), std::move(operand));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildCall(const ValueRecord &value) {
		std::vector<std::unique_ptr<Node::Node>> arguments;
		arguments.reserve(value.arguments.size());
		for (const auto argument : value.arguments) {
			auto node = BuildNode(argument);
			if (node == nullptr) {
				return nullptr;
			}

			arguments.push_back(std::move(node));
		}

		return value.intrinsic.empty()
			? nullptr
			: std::make_unique<Node::CallNode>(value.intrinsic, std::move(arguments));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildSwizzle(const ValueRecord &value) {
		auto vector = BuildNode(value.left);
		if (vector == nullptr || value.member.empty()) {
			return nullptr;
		}

		std::unique_ptr<Node::Node> member = std::make_unique<Node::LoadUniformNode>(value.member);
		return std::make_unique<Node::MemberAccessNode>(vector, member);
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildIndexAccess(const ValueRecord &value) {
		auto instance = BuildNode(value.left);
		auto index	  = BuildNode(value.index);
		if (instance == nullptr || index == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::ArrayAccessNode>(std::move(instance), std::move(index));
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildMemberAccess(const ValueRecord &value) {
		auto instance = BuildNode(value.left);
		if (instance == nullptr || value.member.empty()) {
			return nullptr;
		}

		std::unique_ptr<Node::Node> member = std::make_unique<Node::LoadUniformNode>(value.member);
		return std::make_unique<Node::MemberAccessNode>(instance, member);
	}

	[[nodiscard]] std::unique_ptr<Node::Node> BuildSharedMemoryElement(const ValueRecord &value) {
		auto target = std::make_unique<Node::LoadLocalVariableNode>(value.localName);
		auto index	= BuildNode(value.index);
		if (value.localName.empty() || index == nullptr) {
			return nullptr;
		}

		return std::make_unique<Node::ArrayAccessNode>(std::move(target), std::move(index));
	}

	const Module &_module;
	GPU::AD::GradientTape* _gradientTape = nullptr;
	std::unordered_map<ResourceId, std::string> _uniformNames;
	const std::vector<Block> *_activeBlocks = nullptr;
};

} // namespace

FunctionId ModuleBuilder::BeginComputeKernel(uint32_t workSizeX, uint32_t workSizeY, uint32_t workSizeZ,
											 uint32_t dimension, std::string entryPoint) {
	Function function;
	function.id		   = static_cast<FunctionId>(_module.functions.size());
	function.stage	   = ShaderStage::Compute;
	function.entryPoint = std::move(entryPoint);
	function.workSizeX  = workSizeX;
	function.workSizeY  = workSizeY;
	function.workSizeZ  = workSizeZ;
	function.dimension  = dimension;
	_module.functions.push_back(std::move(function));
	_activeFunction = _module.functions.back().id;
	return _activeFunction;
}

ResourceId ModuleBuilder::AddBuffer(uint32_t binding, Type elementType, ResourceAccess access, std::string name) {
	ResourceBinding resource;
	resource.id			= static_cast<ResourceId>(_module.resources.size());
	resource.binding	 = binding;
	resource.kind		 = ResourceKind::Buffer;
	resource.access		 = access;
	resource.elementType = elementType;
	resource.name		 = std::move(name);
	_module.resources.push_back(std::move(resource));
	return _module.resources.back().id;
}

ResourceId ModuleBuilder::AddTexture2D(uint32_t binding, Type elementType, ResourceAccess access, std::string name,
										Runtime::PixelFormat format, uint32_t width, uint32_t height, bool sampled) {
	ResourceBinding resource;
	resource.id			  = static_cast<ResourceId>(_module.resources.size());
	resource.binding	  = binding;
	resource.kind		  = ResourceKind::Texture;
	resource.access		  = access;
	resource.elementType  = elementType;
	resource.name		  = std::move(name);
	resource.textureFormat = format;
	resource.width		  = width;
	resource.height		  = height;
	resource.depth		  = 1;
	resource.textureDimension = 2;
	resource.sampled	  = sampled;
	_module.resources.push_back(std::move(resource));
	return _module.resources.back().id;
}

ResourceId ModuleBuilder::AddTexture3D(uint32_t binding, Type elementType, ResourceAccess access, std::string name,
										Runtime::PixelFormat format, uint32_t width, uint32_t height, uint32_t depth,
										bool sampled) {
	ResourceBinding resource;
	resource.id			  = static_cast<ResourceId>(_module.resources.size());
	resource.binding	  = binding;
	resource.kind		  = ResourceKind::Texture;
	resource.access		  = access;
	resource.elementType  = elementType;
	resource.name		  = std::move(name);
	resource.textureFormat = format;
	resource.width		  = width;
	resource.height		  = height;
	resource.depth		  = depth;
	resource.textureDimension = 3;
	resource.sampled	  = sampled;
	_module.resources.push_back(std::move(resource));
	return _module.resources.back().id;
}

ResourceId ModuleBuilder::AddPushConstant(uint32_t binding, Type elementType, std::string name, void *data, size_t size,
										   size_t alignment) {
	ResourceBinding resource;
	resource.id			= static_cast<ResourceId>(_module.resources.size());
	resource.binding	 = binding;
	resource.kind		 = ResourceKind::PushConstant;
	resource.access		 = ResourceAccess::Read;
	resource.elementType = elementType;
	resource.name		 = std::move(name);
	resource.data		 = data;
	resource.size		 = size;
	resource.alignment	 = alignment;
	_module.resources.push_back(std::move(resource));
	return _module.resources.back().id;
}

ValueId ModuleBuilder::AddBuiltinValue(ValueRecord::Kind kind) {
	ValueRecord value;
	value.kind = kind;
	value.type = Type::Int();
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::ThreadIndexX() {
	return AddBuiltinValue(ValueRecord::Kind::ThreadIndexX);
}

ValueId ModuleBuilder::ThreadIndexY() {
	return AddBuiltinValue(ValueRecord::Kind::ThreadIndexY);
}

ValueId ModuleBuilder::ThreadIndexZ() {
	return AddBuiltinValue(ValueRecord::Kind::ThreadIndexZ);
}

ValueId ModuleBuilder::LocalIndexX() {
	return AddBuiltinValue(ValueRecord::Kind::LocalIndexX);
}

ValueId ModuleBuilder::LocalIndexY() {
	return AddBuiltinValue(ValueRecord::Kind::LocalIndexY);
}

ValueId ModuleBuilder::LocalIndexZ() {
	return AddBuiltinValue(ValueRecord::Kind::LocalIndexZ);
}

ValueId ModuleBuilder::GroupIdX() {
	return AddBuiltinValue(ValueRecord::Kind::GroupIdX);
}

ValueId ModuleBuilder::GroupIdY() {
	return AddBuiltinValue(ValueRecord::Kind::GroupIdY);
}

ValueId ModuleBuilder::GroupIdZ() {
	return AddBuiltinValue(ValueRecord::Kind::GroupIdZ);
}

ValueId ModuleBuilder::DispatchSizeX() {
	return AddBuiltinValue(ValueRecord::Kind::DispatchSizeX);
}

ValueId ModuleBuilder::DispatchSizeY() {
	return AddBuiltinValue(ValueRecord::Kind::DispatchSizeY);
}

ValueId ModuleBuilder::DispatchSizeZ() {
	return AddBuiltinValue(ValueRecord::Kind::DispatchSizeZ);
}

ValueId ModuleBuilder::GroupSizeX() {
	return AddBuiltinValue(ValueRecord::Kind::GroupSizeX);
}

ValueId ModuleBuilder::GroupSizeY() {
	return AddBuiltinValue(ValueRecord::Kind::GroupSizeY);
}

ValueId ModuleBuilder::GroupSizeZ() {
	return AddBuiltinValue(ValueRecord::Kind::GroupSizeZ);
}

ValueId ModuleBuilder::ResourceElement(ResourceId resource, ValueId index) {
	if (resource >= _module.resources.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::ResourceElement;
	value.type	   = _module.resources[resource].elementType;
	value.resource = resource;
	value.index	   = index;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::TextureElement(ResourceId resource, ValueId x, ValueId y) {
	if (resource >= _module.resources.size() || _module.resources[resource].kind != ResourceKind::Texture ||
		x >= _module.values.size() || y >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::TextureElement;
	value.type	   = _module.resources[resource].elementType;
	value.resource = resource;
	value.index	   = x;
	value.y		   = y;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::TextureElement3D(ResourceId resource, ValueId x, ValueId y, ValueId z) {
	if (resource >= _module.resources.size() || _module.resources[resource].kind != ResourceKind::Texture ||
		x >= _module.values.size() || y >= _module.values.size() || z >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::TextureElement;
	value.type	   = _module.resources[resource].elementType;
	value.resource = resource;
	value.index	   = x;
	value.y		   = y;
	value.right	   = z;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::PushConstant(ResourceId resource) {
	if (resource >= _module.resources.size() || _module.resources[resource].kind != ResourceKind::PushConstant) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::PushConstant;
	value.type	   = _module.resources[resource].elementType;
	value.resource = resource;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Literal(Type type, std::string valueText) {
	ValueRecord value;
	value.kind	  = ValueRecord::Kind::Literal;
	value.type	  = type;
	value.literal = std::move(valueText);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::LocalVariable(Type type, std::string name) {
	ValueRecord value;
	value.kind		= ValueRecord::Kind::LocalVar;
	value.type		= type;
	value.localName = std::move(name);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Ternary(ValueId condition, ValueId trueValue, ValueId falseValue) {
	if (condition >= _module.values.size() || trueValue >= _module.values.size() || falseValue >= _module.values.size())
		return InvalidValueId;
	ValueRecord value;
	value.kind = ValueRecord::Kind::Ternary;
	value.type = _module.values[trueValue].type;
	value.left = condition;
	value.right = trueValue;
	value.arguments.push_back(falseValue);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Binary(BinaryOp op, ValueId left, ValueId right) {
	if (left >= _module.values.size() || right >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::Binary;
	value.type	   = _module.values[left].type;
	value.binaryOp = op;
	value.left	   = left;
	value.right	   = right;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Unary(UnaryOp op, ValueId operand) {
	if (operand >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	  = ValueRecord::Kind::Unary;
	value.type	  = _module.values[operand].type;
	value.unaryOp = op;
	value.left	  = operand;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Intrinsic(std::string name, Type resultType, std::span<const ValueId> arguments) {
	for (const auto argument : arguments) {
		if (argument >= _module.values.size()) {
			return InvalidValueId;
		}
	}

	ValueRecord value;
	value.kind		= ValueRecord::Kind::Intrinsic;
	value.type		= resultType;
	value.intrinsic = std::move(name);
	value.arguments.assign(arguments.begin(), arguments.end());
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::TextureSample(ResourceId resource, Type resultType, ValueId uv) {
	if (resource >= _module.resources.size() || _module.resources[resource].kind != ResourceKind::Texture ||
		!_module.resources[resource].sampled || !resultType.IsValid() || uv >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::TextureSample;
	value.type	   = std::move(resultType);
	value.resource = resource;
	value.arguments.push_back(uv);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::TextureSampleLevel(ResourceId resource, Type resultType, ValueId uv, ValueId lod) {
	if (resource >= _module.resources.size() || _module.resources[resource].kind != ResourceKind::Texture ||
		!_module.resources[resource].sampled || !resultType.IsValid() ||
		uv >= _module.values.size() || lod >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	   = ValueRecord::Kind::TextureSampleLevel;
	value.type	   = std::move(resultType);
	value.resource = resource;
	value.arguments.push_back(uv);
	value.arguments.push_back(lod);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Call(std::string name, Type resultType, std::span<const ValueId> arguments) {
	if (name.empty() || !resultType.IsValid()) {
		return InvalidValueId;
	}
	for (const auto argument : arguments) {
		if (argument >= _module.values.size()) {
			return InvalidValueId;
		}
	}

	ValueRecord value;
	value.kind = ValueRecord::Kind::Call;
	value.type = std::move(resultType);
	value.intrinsic = std::move(name);
	value.arguments.assign(arguments.begin(), arguments.end());
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Atomic(AtomicOp op, Type resultType, ValueId target, std::span<const ValueId> arguments) {
	if (!resultType.IsValid() || target >= _module.values.size() ||
		arguments.empty() || arguments.size() > 2) {
		return InvalidValueId;
	}

	for (const auto argument : arguments) {
		if (argument >= _module.values.size()) {
			return InvalidValueId;
		}
	}

	if ((op == AtomicOp::CompareExchange && arguments.size() != 2) ||
		(op != AtomicOp::CompareExchange && arguments.size() != 1)) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind = ValueRecord::Kind::Atomic;
	value.type = std::move(resultType);
	value.atomicOp = op;
	value.left = target;
	value.arguments.assign(arguments.begin(), arguments.end());
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::Swizzle(ValueId vector, Type resultType, std::string components) {
	if (vector >= _module.values.size() || !resultType.IsValid() || components.empty()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	 = ValueRecord::Kind::Swizzle;
	value.type	 = resultType;
	value.left	 = vector;
	value.member = std::move(components);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::IndexAccess(ValueId instance, ValueId index, Type resultType) {
	if (instance >= _module.values.size() || index >= _module.values.size() || !resultType.IsValid()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind  = ValueRecord::Kind::IndexAccess;
	value.type  = std::move(resultType);
	value.left  = instance;
	value.index = index;
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::MemberAccess(ValueId instance, Type resultType, std::string member) {
	if (instance >= _module.values.size() || !resultType.IsValid() || member.empty()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind	 = ValueRecord::Kind::MemberAccess;
	value.type	 = std::move(resultType);
	value.left	 = instance;
	value.member = std::move(member);
	return AddValue(std::move(value));
}

ValueId ModuleBuilder::SharedMemoryElement(Type elementType, std::string name, ValueId index) {
	if (!elementType.IsValid() || name.empty() || index >= _module.values.size()) {
		return InvalidValueId;
	}

	ValueRecord value;
	value.kind		= ValueRecord::Kind::SharedMemoryElement;
	value.type		= std::move(elementType);
	value.localName = std::move(name);
	value.index		= index;
	return AddValue(std::move(value));
}

FunctionId ModuleBuilder::AddCallable(std::string name, Type returnType,
									   std::vector<std::pair<std::string, Type>> parameters,
									   std::vector<Statement> statements, std::vector<Block> blocks) {
	if (name.empty() || !returnType.IsValid()) {
		return InvalidFunctionId;
	}

	for (const auto &parameter : parameters) {
		if (parameter.first.empty() || !parameter.second.IsValid()) {
			return InvalidFunctionId;
		}
	}

	CallableFunction callable;
	callable.id = static_cast<FunctionId>(_module.callables.size());
	callable.name = std::move(name);
	callable.returnType = std::move(returnType);
	callable.parameters = std::move(parameters);
	callable.statements = std::move(statements);
	callable.blocks = std::move(blocks);
	_module.callables.push_back(std::move(callable));
	return _module.callables.back().id;
}

void ModuleBuilder::DeclareLocal(Type type, std::string name, ValueId initializer) {
	Statement statement;
	statement.kind = Statement::Kind::LocalDeclaration;
	statement.localType = type;
	statement.localName = std::move(name);
	statement.initializer = initializer;
	ActiveFunction().statements.push_back(statement);
}

void ModuleBuilder::Store(ValueId target, ValueId value) {
	Statement statement;
	statement.kind	= Statement::Kind::Store;
	statement.target = target;
	statement.value	= value;
	ActiveFunction().statements.push_back(statement);
}

void ModuleBuilder::RawGLSL(std::string code) {
	// Legacy compatibility escape hatch. New normal DSL features should add
	// typed values/statements and EasyGPU nodes instead of entering here.
	Statement statement;
	statement.kind = Statement::Kind::RawGLSL;
	statement.rawGlsl = std::move(code);
	ActiveFunction().statements.push_back(statement);
}

ValueId ModuleBuilder::Compare(CompareOp op, ValueId left, ValueId right) {
	if (left >= _module.values.size() || right >= _module.values.size()) return InvalidValueId;
	ValueRecord value;
	value.kind	   = ValueRecord::Kind::Compare;
	value.type	   = Type::Bool();
	value.compareOp = op;
	value.left	   = left;
	value.right	   = right;
	return AddValue(std::move(value));
}

BlockId ModuleBuilder::AddBlock(std::vector<Statement> statements) {
	Block block;
	block.id = static_cast<BlockId>(ActiveFunction().blocks.size());
	block.statements = std::move(statements);
	ActiveFunction().blocks.push_back(std::move(block));
	return ActiveFunction().blocks.back().id;
}

void ModuleBuilder::If(ValueId condition, BlockId thenBlock, BlockId elseBlock) {
	Statement statement;
	statement.kind	   = Statement::Kind::If;
	statement.condition = condition;
	statement.thenBlock = thenBlock;
	statement.elseBlock = elseBlock;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::For(BlockId init, ValueId condition, BlockId step, BlockId body) {
	Statement statement;
	statement.kind = Statement::Kind::For;
	statement.initBlock = init;
	statement.condition = condition;
	statement.stepBlock = step;
	statement.bodyBlock = body;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::While(ValueId condition, BlockId body) {
	Statement statement;
	statement.kind	    = Statement::Kind::While;
	statement.condition = condition;
	statement.bodyBlock = body;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::DoWhile(BlockId body, ValueId condition) {
	Statement statement;
	statement.kind	    = Statement::Kind::DoWhile;
	statement.bodyBlock = body;
	statement.condition = condition;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::Break() {
	Statement statement;
	statement.kind = Statement::Kind::Break;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::Continue() {
	Statement statement;
	statement.kind = Statement::Kind::Continue;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::Return(ValueId value) {
	Statement statement;
	statement.kind  = Statement::Kind::Return;
	statement.value = value;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::Expression(ValueId value) {
	Statement statement;
	statement.kind = Statement::Kind::Expression;
	statement.value = value;
	ActiveFunction().statements.push_back(std::move(statement));
}

void ModuleBuilder::Barrier(BarrierKind kind) {
	Statement statement;
	statement.kind = Statement::Kind::Barrier;
	statement.barrierKind = kind;
	ActiveFunction().statements.push_back(statement);
}

void ModuleBuilder::SharedMemoryDecl(Type type, uint32_t count, std::string name) {
	Statement statement;
	statement.kind	   = Statement::Kind::SharedMemoryDecl;
	statement.sharedType = type;
	statement.sharedCount = count;
	statement.sharedName  = std::move(name);
	ActiveFunction().statements.push_back(statement);
}


ValueId ModuleBuilder::AddValue(ValueRecord value) {
	value.id = static_cast<ValueId>(_module.values.size());
	_module.values.push_back(std::move(value));
	return _module.values.back().id;
}

Function &ModuleBuilder::ActiveFunction() {
	if (_activeFunction >= _module.functions.size()) {
		throw std::runtime_error("EasyGPU ModuleBuilder requires an active function.");
	}

	return _module.functions[_activeFunction];
}

std::unique_ptr<Kernel::KernelBuildContext> BuildKernelBuildContext(const Module &module, GPU::AD::GradientTape* tape) {
	// Dispatch to the appropriate build context based on shader stage.
	if (module.functions.empty()) {
		return nullptr;
	}

	const auto& stage = module.functions.front().stage;
	if (stage == ShaderStage::Compute) {
		return ModuleLowerer(module, tape).Build();
	}

	// For vertex/fragment shaders, create a 1D context (each vertex is a thread).
	// Full graphics pipeline support requires a GraphicsBuildContext.
	// For now, use compute context as a fallback.
	return ModuleLowerer(module, tape).Build();
}


} // namespace GPU::IR
