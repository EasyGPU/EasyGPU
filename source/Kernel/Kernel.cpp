/**
 * @file Kernel.cpp
 * @brief Kernel implementation with backend abstraction.
 */

#include <Kernel/Kernel.h>
#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>
#include <Kernel/ShaderCache.h>

#include <IR/Value/Var.h>
#include <IR/Value/VarArray.h>
#include <Runtime/BufferSlot.h>
#include <Runtime/Context.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/ShaderUtils.h>
#include <Runtime/TextureSlot.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace GPU::Kernel {

/**
 * RAII guard for saving and restoring builder context in Kernel constructors.
 */
using KernelBuilderGuard = IR::Builder::Builder::ScopedBind;

/**
 * RAII guard for backend shader handles during pipeline creation.
 */
class ShaderGuard {
public:
	ShaderGuard(Backend::Backend &backend, Backend::ShaderHandle shader) : _backend(backend), _shader(shader) {
	}

	~ShaderGuard() {
		if (_shader != Backend::INVALID_SHADER_HANDLE) {
			_backend.DestroyShader(_shader);
		}
	}

	ShaderGuard(const ShaderGuard &)			= delete;
	ShaderGuard &operator=(const ShaderGuard &) = delete;
	ShaderGuard(ShaderGuard &&)					= delete;
	ShaderGuard &operator=(ShaderGuard &&)		= delete;

private:
	Backend::Backend	 &_backend;
	Backend::ShaderHandle _shader;
};

// ===================================================================================
// KernelBase - Common functionality
// ===================================================================================

void KernelBase::WorkgroupBarrier() {
	auto *context = IR::Builder::Builder::Get().Context();
	if (context != nullptr) {
		context->PushTranslatedCode("barrier();\n");
	}
}

void KernelBase::MemoryBarrier() {
	auto *context = IR::Builder::Builder::Get().Context();
	if (context != nullptr) {
		context->PushTranslatedCode("memoryBarrier();\n");
	}
}

void KernelBase::FullBarrier() {
	auto *context = IR::Builder::Builder::Get().Context();
	if (context != nullptr) {
		context->PushTranslatedCode("memoryBarrier();\n");
		context->PushTranslatedCode("barrier();\n");
	}
}

void KernelBase::RuntimeBarrier() {
	auto *backend = Runtime::Context::GetBackend();
	if (backend) {
		backend->MemoryBarrier(Backend::BarrierType::All);
	}
}

// ===================================================================================
// Internal dispatch helper
// ===================================================================================

/**
 * Execute the compute shader dispatch using the backend
 */
static void ExecuteComputeDispatch(KernelBuildContext &context, int groupX, int groupY, int groupZ, bool sync = false) {
	// Initialize context
	Runtime::AutoInitContext();

	// Make context current
	Runtime::Context::GetInstance().MakeCurrent();

	// Get the backend
	auto *backend = Runtime::Context::GetBackend();
	if (!backend) {
		throw std::runtime_error("Backend not available");
	}

	// Compute shader hash for cache lookup (if not already computed)
	if (context.GetShaderHash().empty()) {
		context.ComputeShaderHash();
	}

	// Get or create the cached pipeline
	Backend::PipelineHandle pipeline = context.GetCachedPipeline();
	if (pipeline == Backend::INVALID_PIPELINE_HANDLE) {
		// Get the complete shader code
		std::string			shaderSource = context.GetCompleteCode();

		// Vulkan pipeline cache data accelerates pipeline creation but does not
		// replace the shader module required by vkCreateComputePipelines.
		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = shaderSource;
		shaderDesc.entryPoint		 = "main";
		shaderDesc.optimizationLevel = context.GetOptimizationLevel();

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE) {
			throw std::runtime_error("Failed to create compute shader");
		}
		ShaderGuard shaderGuard(*backend, shader);

		// Try to load from binary cache first
		bool		loadedFromCache = false;
		if (backend->SupportsPipelineCache()) {
			auto &globalCache = GlobalShaderCache::Get();
			auto  entry = globalCache.Lookup(context.GetShaderHash(),
											 static_cast<uint32_t>(Runtime::Context::GetInstance().GetBackendType()));

			if (entry && entry->dataSize > 0) {
				// Try to create pipeline from cached binary
				Backend::PipelineDesc pipelineDesc;
				pipelineDesc.computeShader	  = shader;
				pipelineDesc.workGroupSizeX	  = context.WorkSizeX;
				pipelineDesc.workGroupSizeY	  = context.WorkSizeY;
				pipelineDesc.workGroupSizeZ	  = context.WorkSizeZ;
				pipelineDesc.pushConstantSize = context.GetPushConstantSize();

				for (const auto &bufferInfo : context.GetBufferInfos()) {
					Backend::ResourceLayoutEntry entryLayout;
					entryLayout.binding	 = bufferInfo.binding;
					entryLayout.type	 = Backend::BindingType::Buffer;
					entryLayout.readOnly = (bufferInfo.mode == GPU::Backend::BUFFER_MODE_READ_ONLY);
					pipelineDesc.resources.push_back(entryLayout);
				}

				for (const auto &textureInfo : context.GetTextureInfos()) {
					Backend::ResourceLayoutEntry entryLayout;
					entryLayout.binding = textureInfo.binding;
					entryLayout.type =
						textureInfo.sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
					entryLayout.format	 = Runtime::ToBackendPixelFormat(textureInfo.format);
					entryLayout.readOnly = textureInfo.sampled;
					pipelineDesc.resources.push_back(entryLayout);
				}

				pipeline = backend->CreatePipelineFromBinary(pipelineDesc, entry->data.data(), entry->data.size(),
															 entry->format);

				if (pipeline != Backend::INVALID_PIPELINE_HANDLE) {
					loadedFromCache = true;
					context.SetCachedBinaryFormat(entry->format);
				}
			}
		}

		// If not loaded from cache, compile from source
		if (!loadedFromCache) {
			// Create pipeline
			Backend::PipelineDesc pipelineDesc;
			pipelineDesc.computeShader	  = shader;
			pipelineDesc.workGroupSizeX	  = context.WorkSizeX;
			pipelineDesc.workGroupSizeY	  = context.WorkSizeY;
			pipelineDesc.workGroupSizeZ	  = context.WorkSizeZ;
			pipelineDesc.pushConstantSize = context.GetPushConstantSize();

			for (const auto &bufferInfo : context.GetBufferInfos()) {
				Backend::ResourceLayoutEntry entry;
				entry.binding  = bufferInfo.binding;
				entry.type	   = Backend::BindingType::Buffer;
				entry.readOnly = (bufferInfo.mode == GPU::Backend::BUFFER_MODE_READ_ONLY);
				pipelineDesc.resources.push_back(entry);
			}

			for (const auto &textureInfo : context.GetTextureInfos()) {
				Backend::ResourceLayoutEntry entry;
				entry.binding  = textureInfo.binding;
				entry.type	   = textureInfo.sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
				entry.format   = Runtime::ToBackendPixelFormat(textureInfo.format);
				entry.readOnly = textureInfo.sampled;
				pipelineDesc.resources.push_back(entry);
			}

			pipeline = backend->CreatePipeline(pipelineDesc);
			if (pipeline == Backend::INVALID_PIPELINE_HANDLE) {
				throw std::runtime_error("Failed to create compute pipeline");
			}

			// Cache the binary for future use
			if (backend->SupportsPipelineCache()) {
				uint32_t format = 0;
				auto	 binary = backend->GetPipelineBinary(pipeline, format);
				if (!binary.empty()) {
					auto &globalCache = GlobalShaderCache::Get();
					globalCache.Store(context.GetShaderHash(),
									  static_cast<uint32_t>(Runtime::Context::GetInstance().GetBackendType()), format,
									  binary);
					context.SetCachedBinaryFormat(format);
				}
			}
		}

		// Cache the pipeline handle for future dispatches in this session
		context.SetCachedPipeline(pipeline);
	}

	// Bind the pipeline
	backend->BindPipeline(pipeline);

	// Upload uniform values
	context.UploadUniformValues(pipeline);

	// Prepare resource bindings.
	const auto &bindings = context.GetCachedBindings();

	// Bind all resources
	if (!bindings.empty()) {
		backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
	}

	// Dispatch the compute shader
	backend->Dispatch(groupX, groupY, groupZ);

	// Apply pre-computed memory barriers for writable resources.
	Backend::BarrierType barrierType = context.GetRequiredBarrierType();
	if (barrierType != Backend::BarrierType::None) {
		backend->MemoryBarrier(barrierType);
	}

	// Sync if requested
	if (sync) {
		backend->Finish();
	}
}

// ===================================================================================
// Inspector Kernels - For debugging
// ===================================================================================

/**
 * Shared helper for compiling shader code from a build context
 */
static bool CompileShaderInternal(KernelBuildContext &context, std::string &errorMessage) {
	try {
		Runtime::AutoInitContext();
		Runtime::ContextGuard guard(Runtime::Context::GetInstance());

		std::string			  shaderSource = context.GetCompleteCode();

		auto				 *backend	   = Runtime::Context::GetBackend();
		if (!backend) {
			errorMessage = "Backend not available";
			return false;
		}

		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = shaderSource;
		shaderDesc.optimizationLevel = context.GetOptimizationLevel();

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE) {
			errorMessage = "Shader compilation failed";
			return false;
		}

		backend->DestroyShader(shader);
		return true;
	} catch (const std::exception &e) {
		errorMessage = e.what();
		return false;
	}
}

static std::string GetOptimizedGLSLInternal(KernelBuildContext &context) {
	Runtime::AutoInitContext();
	Runtime::ContextGuard guard(Runtime::Context::GetInstance());

	auto *backend = Runtime::Context::GetBackend();
	if (!backend) {
		throw std::runtime_error("Backend not available");
	}

	Backend::ShaderDesc shaderDesc;
	shaderDesc.type				 = Backend::ShaderType::Compute;
	shaderDesc.sourceCode		 = context.GetCompleteCode();
	shaderDesc.entryPoint		 = "main";
	shaderDesc.optimizationLevel = context.GetOptimizationLevel();

	return backend->GetOptimizedGLSL(shaderDesc);
}

InspectorKernel1D::InspectorKernel1D(const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX)
	: _context(1) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;

	IR::Value::Var<int> Id("(int(gl_GlobalInvocationID.x))");
	Func(Id);
}

void InspectorKernel1D::PrintCode() {
	std::cout << _context.GetCompleteCode() << std::endl;
}

std::string InspectorKernel1D::GetCode() {
	return _context.GetCompleteCode();
}

std::string InspectorKernel1D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void InspectorKernel1D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel InspectorKernel1D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

bool InspectorKernel1D::Compile() {
	std::string unused;
	return Compile(unused);
}

bool InspectorKernel1D::Compile(std::string &errorMessage) {
	try {
		Runtime::AutoInitContext();
		Runtime::ContextGuard guard(Runtime::Context::GetInstance());

		std::string			  shaderSource = _context.GetCompleteCode();

		auto				 *backend	   = Runtime::Context::GetBackend();
		if (!backend) {
			errorMessage = "Backend not available";
			return false;
		}

		// Create and compile shader through backend
		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = shaderSource;
		shaderDesc.optimizationLevel = _context.GetOptimizationLevel();

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE) {
			errorMessage = "Shader compilation failed";
			return false;
		}

		backend->DestroyShader(shader);
		return true;
	} catch (const std::exception &e) {
		errorMessage = e.what();
		return false;
	}
}

InspectorKernel2D::InspectorKernel2D(
	const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func, int WorkSizeX, int WorkSizeY)
	: _context(2) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	Func(IdX, IdY);
}

void InspectorKernel2D::PrintCode() {
	std::cout << _context.GetCompleteCode() << std::endl;
}

std::string InspectorKernel2D::GetCode() {
	return _context.GetCompleteCode();
}

std::string InspectorKernel2D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void InspectorKernel2D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel InspectorKernel2D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

bool InspectorKernel2D::Compile() {
	std::string unused;
	return Compile(unused);
}

bool InspectorKernel2D::Compile(std::string &errorMessage) {
	try {
		Runtime::AutoInitContext();
		Runtime::ContextGuard guard(Runtime::Context::GetInstance());

		std::string			  shaderSource = _context.GetCompleteCode();

		auto				 *backend	   = Runtime::Context::GetBackend();
		if (!backend) {
			errorMessage = "Backend not available";
			return false;
		}

		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = shaderSource;
		shaderDesc.optimizationLevel = _context.GetOptimizationLevel();

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE) {
			errorMessage = "Shader compilation failed";
			return false;
		}

		backend->DestroyShader(shader);
		return true;
	} catch (const std::exception &e) {
		errorMessage = e.what();
		return false;
	}
}

InspectorKernel3D::InspectorKernel3D(
	const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
	int WorkSizeX, int WorkSizeY, int WorkSizeZ)
	: _context(3) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;
	_context.WorkSizeZ = WorkSizeZ;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	IR::Value::Var<int> IdZ("(int(gl_GlobalInvocationID.z))");
	Func(IdX, IdY, IdZ);
}

void InspectorKernel3D::PrintCode() {
	std::cout << _context.GetCompleteCode() << std::endl;
}

std::string InspectorKernel3D::GetCode() {
	return _context.GetCompleteCode();
}

std::string InspectorKernel3D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void InspectorKernel3D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel InspectorKernel3D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

bool InspectorKernel3D::Compile() {
	std::string unused;
	return Compile(unused);
}

bool InspectorKernel3D::Compile(std::string &errorMessage) {
	try {
		Runtime::AutoInitContext();
		Runtime::ContextGuard guard(Runtime::Context::GetInstance());

		std::string			  shaderSource = _context.GetCompleteCode();

		auto				 *backend	   = Runtime::Context::GetBackend();
		if (!backend) {
			errorMessage = "Backend not available";
			return false;
		}

		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = shaderSource;
		shaderDesc.optimizationLevel = _context.GetOptimizationLevel();

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE) {
			errorMessage = "Shader compilation failed";
			return false;
		}

		backend->DestroyShader(shader);
		return true;
	} catch (const std::exception &e) {
		errorMessage = e.what();
		return false;
	}
}

// ===================================================================================
// Executable Kernels
// ===================================================================================

Kernel1D::Kernel1D(const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX)
	: _context(1), _name("Kernel1D") {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;

	IR::Value::Var<int> Id("(int(gl_GlobalInvocationID.x))");
	Func(Id);
}

Kernel1D::Kernel1D(const std::string &name, const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX)
	: _context(1), _name(name) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;

	IR::Value::Var<int> Id("(int(gl_GlobalInvocationID.x))");
	Func(Id);
}

void Kernel1D::SetName(const std::string &name) {
	_name = name;
}

std::string Kernel1D::GetName() const {
	return _name;
}

void Kernel1D::Dispatch(int GroupX, bool sync) {
	auto &profiler = KernelProfiler::GetInstance();
	if (profiler.IsEnabled()) {
		unsigned int queryId = profiler.BeginQuery();
		ExecuteComputeDispatch(_context, GroupX, 1, 1, sync && queryId == 0);
		profiler.EndQuery(queryId, _name, GroupX, 1, 1);
	} else {
		ExecuteComputeDispatch(_context, GroupX, 1, 1, sync);
	}
}

std::string Kernel1D::GetCode() {
	return _context.GetCompleteCode();
}

std::string Kernel1D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void Kernel1D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel Kernel1D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

Kernel2D::Kernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func, int WorkSizeX,
				   int WorkSizeY)
	: _context(2), _name("Kernel2D") {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	Func(IdX, IdY);
}

Kernel2D::Kernel2D(const std::string															 &name,
				   const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func, int WorkSizeX,
				   int WorkSizeY)
	: _context(2), _name(name) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	Func(IdX, IdY);
}

void Kernel2D::SetName(const std::string &name) {
	_name = name;
}

std::string Kernel2D::GetName() const {
	return _name;
}

void Kernel2D::Dispatch(int GroupX, int GroupY, bool sync) {
	auto &profiler = KernelProfiler::GetInstance();
	if (profiler.IsEnabled()) {
		unsigned int queryId = profiler.BeginQuery();
		ExecuteComputeDispatch(_context, GroupX, GroupY, 1, sync && queryId == 0);
		profiler.EndQuery(queryId, _name, GroupX, GroupY, 1);
	} else {
		ExecuteComputeDispatch(_context, GroupX, GroupY, 1, sync);
	}
}

std::string Kernel2D::GetCode() {
	return _context.GetCompleteCode();
}

std::string Kernel2D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void Kernel2D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel Kernel2D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

Kernel3D::Kernel3D(
	const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
	int WorkSizeX, int WorkSizeY, int WorkSizeZ)
	: _context(3), _name("Kernel3D") {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;
	_context.WorkSizeZ = WorkSizeZ;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	IR::Value::Var<int> IdZ("(int(gl_GlobalInvocationID.z))");
	Func(IdX, IdY, IdZ);
}

Kernel3D::Kernel3D(
	const std::string																						&name,
	const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
	int WorkSizeX, int WorkSizeY, int WorkSizeZ)
	: _context(3), _name(name) {
	KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

	_context.WorkSizeX = WorkSizeX;
	_context.WorkSizeY = WorkSizeY;
	_context.WorkSizeZ = WorkSizeZ;

	IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
	IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
	IR::Value::Var<int> IdZ("(int(gl_GlobalInvocationID.z))");
	Func(IdX, IdY, IdZ);
}

void Kernel3D::SetName(const std::string &name) {
	_name = name;
}

std::string Kernel3D::GetName() const {
	return _name;
}

void Kernel3D::Dispatch(int GroupX, int GroupY, int GroupZ, bool sync) {
	auto &profiler = KernelProfiler::GetInstance();
	if (profiler.IsEnabled()) {
		unsigned int queryId = profiler.BeginQuery();
		ExecuteComputeDispatch(_context, GroupX, GroupY, GroupZ, sync && queryId == 0);
		profiler.EndQuery(queryId, _name, GroupX, GroupY, GroupZ);
	} else {
		ExecuteComputeDispatch(_context, GroupX, GroupY, GroupZ, sync);
	}
}

std::string Kernel3D::GetCode() {
	return _context.GetCompleteCode();
}

std::string Kernel3D::GetOptimizedGLSL() {
	return GetOptimizedGLSLInternal(_context);
}

void Kernel3D::SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
	_context.SetOptimizationLevel(level);
}

Backend::ShaderOptimizationLevel Kernel3D::GetOptimizationLevel() const {
	return _context.GetOptimizationLevel();
}

} // namespace GPU::Kernel
