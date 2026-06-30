#pragma once

/**
 * @file Module.h
 * @brief Language-neutral EasyGPU IR module and module builder.
 */

#ifndef EASYGPU_IR_MODULE_H
#define EASYGPU_IR_MODULE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <Runtime/PixelFormat.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace GPU::Kernel {
class KernelBuildContext;
} // namespace GPU::Kernel

namespace GPU::AD {
class GradientTape;
} // namespace GPU::AD

namespace GPU::IR {

using FunctionId = uint32_t;
using ResourceId = uint32_t;
using ValueId = uint32_t;
using BlockId = uint32_t;

inline constexpr FunctionId InvalidFunctionId = UINT32_MAX;
inline constexpr ResourceId InvalidResourceId = UINT32_MAX;
inline constexpr ValueId    InvalidValueId = UINT32_MAX;
inline constexpr BlockId    InvalidBlockId = UINT32_MAX;

/**
 * Shader stage represented by an IR function.
 */
enum class ShaderStage {
	Compute,
	Vertex,
	Fragment,
};

/**
 * Resource category owned by a module.
 */
enum class ResourceKind {
	Buffer,
	Texture,
	Sampler,
	PushConstant,
};

/**
 * Access mode for module resources.
 */
enum class ResourceAccess {
	Read,
	Write,
	ReadWrite,
};

/**
 * Binary arithmetic operation.
 */
enum class CompareOp {
	Less,
	LessEqual,
	Greater,
	GreaterEqual,
	Equal,
	NotEqual,
};

enum class BinaryOp {
	Add,
	Sub,
	Mul,
	Div,
	Mod,
	BitAnd,
	BitOr,
	BitXor,
	ShiftLeft,
	ShiftRight,
	LogicalAnd,
	LogicalOr,
};

enum class UnaryOp {
	Negate,
	LogicalNot,
	BitwiseNot,
};

enum class AtomicOp {
	Add,
	Sub,
	Min,
	Max,
	And,
	Or,
	Xor,
	Exchange,
	CompareExchange,
};

/**
 * Scalar or aggregate value type known to the language-neutral IR.
 */
	struct Type {
	enum class Kind {
		Unknown,
		Void,
		Bool,
		Int,
		UInt,
		Float,
		Bool2,
		Bool3,
		Bool4,
		Int2,
		Int3,
		Int4,
		UInt2,
		UInt3,
		UInt4,
		Float2,
		Float3,
		Float4,
		Float2x2,
		Float3x3,
		Float4x4,
		Struct,
	};

	Kind kind = Kind::Unknown;
	std::string typeName;
	std::string definition;
	std::vector<std::pair<std::string, std::string>> dependencyDefinitions;

	[[nodiscard]] static Type Void() {
		return {Kind::Void};
	}

	[[nodiscard]] static Type Bool() {
		return {Kind::Bool};
	}

	[[nodiscard]] static Type Int() {
		return {Kind::Int};
	}

	[[nodiscard]] static Type UInt() {
		return {Kind::UInt};
	}

	[[nodiscard]] static Type Float() {
		return {Kind::Float};
	}

	[[nodiscard]] static Type Bool2() {
		return {Kind::Bool2};
	}

	[[nodiscard]] static Type Bool3() {
		return {Kind::Bool3};
	}

	[[nodiscard]] static Type Bool4() {
		return {Kind::Bool4};
	}

	[[nodiscard]] static Type Int2() {
		return {Kind::Int2};
	}

	[[nodiscard]] static Type Int3() {
		return {Kind::Int3};
	}

	[[nodiscard]] static Type Int4() {
		return {Kind::Int4};
	}

	[[nodiscard]] static Type UInt2() {
		return {Kind::UInt2};
	}

	[[nodiscard]] static Type UInt3() {
		return {Kind::UInt3};
	}

	[[nodiscard]] static Type UInt4() {
		return {Kind::UInt4};
	}

	[[nodiscard]] static Type Float2() {
		return {Kind::Float2};
	}

	[[nodiscard]] static Type Float3() {
		return {Kind::Float3};
	}

	[[nodiscard]] static Type Float4() {
		return {Kind::Float4};
	}

	[[nodiscard]] static Type Float2x2() {
		return {Kind::Float2x2};
	}

	[[nodiscard]] static Type Float3x3() {
		return {Kind::Float3x3};
	}

	[[nodiscard]] static Type Float4x4() {
		return {Kind::Float4x4};
	}

	[[nodiscard]] static Type Struct(
		std::string name,
		std::string glslDefinition,
		std::vector<std::pair<std::string, std::string>> dependencyDefinitions = {}) {
		Type type;
		type.kind = Kind::Struct;
		type.typeName = std::move(name);
		type.definition = std::move(glslDefinition);
		type.dependencyDefinitions = std::move(dependencyDefinitions);
		return type;
	}

	[[nodiscard]] bool IsValid() const {
		return kind != Kind::Unknown;
	}
};

/**
 * Resource binding metadata for a module.
 */
struct ResourceBinding {
	ResourceId			 id = InvalidResourceId;
	uint32_t			 binding = 0;
	ResourceKind		 kind = ResourceKind::Buffer;
	ResourceAccess		 access = ResourceAccess::ReadWrite;
	Type				 elementType;
	std::string			 name;
	void				*data = nullptr;
	size_t				 size = 0;
	size_t				 alignment = 0;
	Runtime::PixelFormat textureFormat = Runtime::PixelFormat::RGBA8;
	uint32_t			 width = 0;
	uint32_t			 height = 0;
	uint32_t			 depth = 1;
	uint32_t			 textureDimension = 2;
	bool				 sampled = false;
};

/**
 * Value record stored in an EasyGPU IR module.
 */
struct ValueRecord {
	enum class Kind {
		ThreadIndexX,
		ThreadIndexY,
		ThreadIndexZ,
		LocalIndexX,
		LocalIndexY,
		LocalIndexZ,
		GroupIdX,
		GroupIdY,
		GroupIdZ,
		DispatchSizeX,
		DispatchSizeY,
		DispatchSizeZ,
		GroupSizeX,
		GroupSizeY,
		GroupSizeZ,
		Compare,
		ResourceElement,
		TextureElement,
		PushConstant,
		Literal,
		Binary,
		Unary,
		Intrinsic,
		TextureSample,
		TextureSampleLevel,
		Swizzle,
		IndexAccess,
		MemberAccess,
		SharedMemoryElement,
		LocalVar,
		Ternary,
		Call,
		Atomic,
	};

	ValueId				id = InvalidValueId;
	Kind				kind = Kind::Literal;
	Type				type;
	ResourceId			resource = InvalidResourceId;
	ValueId				index = InvalidValueId;
	ValueId				y = InvalidValueId;
	std::string			literal;
	BinaryOp			binaryOp = BinaryOp::Add;
	UnaryOp				unaryOp = UnaryOp::Negate;
	CompareOp			compareOp = CompareOp::Equal;
	AtomicOp			atomicOp = AtomicOp::Add;
	ValueId				left = InvalidValueId;
	ValueId				right = InvalidValueId;
	std::string			intrinsic;
	std::string			member;
	std::vector<ValueId>	arguments;
	std::string			localName;
};

/**
 * Statement stored in an EasyGPU IR function.
 */
enum class BarrierKind : uint8_t {
	Workgroup,
	Memory,
	Full,
};

struct Statement {
	enum class Kind {
		LocalDeclaration,
		Store,
		If,
		For,
		While,
		DoWhile,
		Break,
		Continue,
		Return,
		Expression,
		Barrier,
		SharedMemoryDecl,
		RawGLSL,
	};

	Kind	kind = Kind::Store;
	ValueId target = InvalidValueId;
	ValueId value = InvalidValueId;
	// Control flow
	ValueId condition = InvalidValueId;
	BlockId thenBlock = InvalidBlockId;
	BlockId elseBlock = InvalidBlockId;
	BlockId bodyBlock = InvalidBlockId;
	BlockId initBlock = InvalidBlockId;
	BlockId stepBlock = InvalidBlockId;
	// Local declarations
	Type localType;
	std::string localName;
	ValueId initializer = InvalidValueId;
	// Shared memory
	Type	sharedType;
	uint32_t sharedCount = 0;
	std::string sharedName;
	// Synchronization
	BarrierKind barrierKind = BarrierKind::Workgroup;
	// Raw GLSL injection (non-differentiable ops)
	std::string rawGlsl;
};


/**
 * Block of statements identified by an ID, used for control flow bodies.
 */
struct Block {
	BlockId id = InvalidBlockId;
	std::vector<Statement> statements;
};

/**
 * Callable helper function emitted alongside the shader entry point.
 */
struct CallableFunction {
	FunctionId			   id = InvalidFunctionId;
	std::string			   name;
	Type				   returnType;
	std::vector<std::pair<std::string, Type>> parameters;
	std::vector<Statement> statements;
	std::vector<Block>	   blocks;
};

/**
 * Function body for one shader stage.
 */
struct Function {
	FunctionId			   id = InvalidFunctionId;
	ShaderStage			   stage = ShaderStage::Compute;
	std::string			   entryPoint = "main";
	uint32_t			   workSizeX = 1;
	uint32_t			   workSizeY = 1;
	uint32_t			   workSizeZ = 1;
	uint32_t			   dimension = 1;
	std::vector<Statement> statements;
	std::vector<Block>	   blocks;
};

/**
 * Language-neutral shader module shared by C++, C#, AD, and backend lowering.
 */
struct Module {
	uint32_t					 version = 1;
	std::vector<ResourceBinding> resources;
	std::vector<ValueRecord>		 values;
	std::vector<Function>		 functions;
	std::vector<CallableFunction> callables;
};

/**
 * Builder for the language-neutral EasyGPU IR module.
 */
class ModuleBuilder {
public:
	FunctionId BeginComputeKernel(uint32_t workSizeX, uint32_t workSizeY, uint32_t workSizeZ,
								  uint32_t dimension = 1, std::string entryPoint = "main");

	ResourceId AddBuffer(uint32_t binding, Type elementType, ResourceAccess access, std::string name);

	ResourceId AddTexture2D(uint32_t binding, Type elementType, ResourceAccess access, std::string name,
							Runtime::PixelFormat format, uint32_t width, uint32_t height, bool sampled = false);

	ResourceId AddTexture3D(uint32_t binding, Type elementType, ResourceAccess access, std::string name,
							Runtime::PixelFormat format, uint32_t width, uint32_t height, uint32_t depth,
							bool sampled = false);

	ResourceId AddPushConstant(uint32_t binding, Type elementType, std::string name, void *data, size_t size,
							   size_t alignment);

	ValueId ThreadIndexX();
	ValueId ThreadIndexY();
	ValueId ThreadIndexZ();
	ValueId LocalIndexX();
	ValueId LocalIndexY();
	ValueId LocalIndexZ();
	ValueId GroupIdX();
	ValueId GroupIdY();
	ValueId GroupIdZ();
	ValueId DispatchSizeX();
	ValueId DispatchSizeY();
	ValueId DispatchSizeZ();
	ValueId GroupSizeX();
	ValueId GroupSizeY();
	ValueId GroupSizeZ();
	ValueId ResourceElement(ResourceId resource, ValueId index);
	ValueId TextureElement(ResourceId resource, ValueId x, ValueId y);
	ValueId TextureElement3D(ResourceId resource, ValueId x, ValueId y, ValueId z);
	ValueId PushConstant(ResourceId resource);
	ValueId Literal(Type type, std::string value);
	ValueId LocalVariable(Type type, std::string name);
		ValueId Ternary(ValueId condition, ValueId trueValue, ValueId falseValue);
	ValueId Binary(BinaryOp op, ValueId left, ValueId right);
	ValueId Unary(UnaryOp op, ValueId operand);
	ValueId Intrinsic(std::string name, Type resultType, std::span<const ValueId> arguments);
	ValueId TextureSample(ResourceId resource, Type resultType, ValueId uv);
	ValueId TextureSampleLevel(ResourceId resource, Type resultType, ValueId uv, ValueId lod);
	ValueId Call(std::string name, Type resultType, std::span<const ValueId> arguments);
	ValueId Atomic(AtomicOp op, Type resultType, ValueId target, std::span<const ValueId> arguments);
	ValueId Swizzle(ValueId vector, Type resultType, std::string components);
	ValueId IndexAccess(ValueId instance, ValueId index, Type resultType);
	ValueId MemberAccess(ValueId instance, Type resultType, std::string member);
	ValueId SharedMemoryElement(Type elementType, std::string name, ValueId index);
	FunctionId AddCallable(std::string name, Type returnType, std::vector<std::pair<std::string, Type>> parameters,
							std::vector<Statement> statements, std::vector<Block> blocks);
	void DeclareLocal(Type type, std::string name, ValueId initializer = InvalidValueId);
	void Store(ValueId target, ValueId value);
	ValueId Compare(CompareOp op, ValueId left, ValueId right);
	BlockId AddBlock(std::vector<Statement> statements);
	void If(ValueId condition, BlockId thenBlock, BlockId elseBlock = InvalidBlockId);
	void For(BlockId init, ValueId condition, BlockId step, BlockId body);
	void While(ValueId condition, BlockId body);
	void DoWhile(BlockId body, ValueId condition);
	void Break();
	void Continue();
	void Return(ValueId value = InvalidValueId);
	void Expression(ValueId value);
	void RawGLSL(std::string code);
	void Barrier(BarrierKind kind = BarrierKind::Workgroup);
	void SharedMemoryDecl(Type type, uint32_t count, std::string name);

	[[nodiscard]] const Module &GetModule() const {
		return _module;
	}

private:
	ValueId AddValue(ValueRecord value);
	ValueId AddBuiltinValue(ValueRecord::Kind kind);
	Function &ActiveFunction();

	Module	   _module;
	FunctionId _activeFunction = InvalidFunctionId;
};

std::unique_ptr<Kernel::KernelBuildContext> BuildKernelBuildContext(const Module &module, GPU::AD::GradientTape* tape = nullptr);

} // namespace GPU::IR

#endif // EASYGPU_IR_MODULE_H
