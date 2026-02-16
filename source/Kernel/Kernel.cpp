/**
 * Kernel.cpp:
 *      @Descripiton    :   The kernel function definition
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#include <Kernel/Kernel.h>
#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>

#include <IR/Value/Var.h>
#include <IR/Value/VarArray.h>
#include <Runtime/Context.h>
#include <Runtime/ShaderUtils.h>

#include <stdexcept>
#include <iostream>

#include <glad/glad.h>

namespace GPU::Kernel {
    /**
     * RAII guard for saving and restoring builder context in Kernel constructors.
     * This ensures correct context isolation when multiple Kernels are constructed
     * in the same thread (even though nested Kernel definitions are not recommended).
     */
    class KernelBuilderGuard {
    public:
        KernelBuilderGuard(IR::Builder::Builder& builder, KernelBuildContext& newContext)
            : _builder(builder)
            , _previousContext(builder.Context()) {
            _builder.Bind(newContext);
        }

        ~KernelBuilderGuard() {
            // Restore previous context (may be nullptr if this is the first Kernel)
            if (_previousContext != nullptr) {
                _builder.Bind(*_previousContext);
            } else {
                _builder.Unbind();
            }
        }

        // Disable copy and move
        KernelBuilderGuard(const KernelBuilderGuard&) = delete;
        KernelBuilderGuard& operator=(const KernelBuilderGuard&) = delete;
        KernelBuilderGuard(KernelBuilderGuard&&) = delete;
        KernelBuilderGuard& operator=(KernelBuilderGuard&&) = delete;

    private:
        IR::Builder::Builder& _builder;
        IR::Builder::BuilderContext* _previousContext;
    };

    // ===================================================================================
    // KernelBase - Common functionality
    // ===================================================================================

    void KernelBase::WorkgroupBarrier() {
        auto* context = IR::Builder::Builder::Get().Context();
        if (context != nullptr) {
            context->PushTranslatedCode("barrier();\n");
        }
    }

    void KernelBase::MemoryBarrier() {
        auto* context = IR::Builder::Builder::Get().Context();
        if (context != nullptr) {
            context->PushTranslatedCode("memoryBarrier();\n");
        }
    }

    void KernelBase::FullBarrier() {
        auto* context = IR::Builder::Builder::Get().Context();
        if (context != nullptr) {
            context->PushTranslatedCode("memoryBarrier();\n");
            context->PushTranslatedCode("barrier();\n");
        }
    }

    void KernelBase::RuntimeBarrier() {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // ===================================================================================
    // Internal dispatch helper
    // ===================================================================================

    /**
     * Execute the OpenGL compute shader dispatch
     * @param context The kernel build context
     * @param groupX The x group count
     * @param groupY The y group count
     * @param groupZ The z group count
     * @param sync Whether to wait for completion
     */
    static void ExecuteComputeDispatch(KernelBuildContext &context, int groupX, int groupY, int groupZ, 
                                       bool sync = false) {
        // Initialize context
        Runtime::AutoInitContext();

        // Create context guard
        Runtime::ContextGuard guard(Runtime::Context::GetInstance());

        // Get the complete shader code
        std::string shaderSource = context.GetCompleteCode();

        // Compile the shader
        uint32_t program = Runtime::ShaderCompiler::CompileComputeShader(shaderSource);

        // Use the program
        glUseProgram(program);

        // Upload uniform values
        context.UploadUniformValues(program);

        // Bind all buffers to their specified binding points
        const auto& bufferBindings = context.GetRuntimeBufferBindings();
        for (const auto& [binding, handle] : bufferBindings) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, handle);
        }

        // Bind all textures to their specified binding points
        const auto& textureBindings = context.GetRuntimeTextureBindings();
        for (const auto& [binding, handle] : textureBindings) {
            // TODO: Format and access mode should be configurable per texture
            glBindImageTexture(binding, handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
        }

        // Dispatch the compute shader
        glDispatchCompute(groupX, groupY, groupZ);

        // Sync if requested
        if (sync) {
            KernelBase::RuntimeBarrier();
        }

        // Unbind buffers
        for (const auto& [binding, handle] : bufferBindings) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, 0);
        }

        // Unbind textures
        for (const auto& [binding, handle] : textureBindings) {
            glBindImageTexture(binding, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
        }

        // Unbind the program
        glUseProgram(0);

        // Delete the program
        glDeleteProgram(program);
    }

    // ===================================================================================
    // Inspector Kernels - For debugging
    // ===================================================================================

    InspectorKernel1D::InspectorKernel1D(const std::function<void(IR::Value::Var<int> &Id)>& Func, int WorkSizeX) 
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

    bool InspectorKernel1D::Compile() {
        std::string unused;
        return Compile(unused);
    }

    bool InspectorKernel1D::Compile(std::string& errorMessage) {
        try {
            Runtime::AutoInitContext();
            Runtime::ContextGuard guard(Runtime::Context::GetInstance());
            
            std::string shaderSource = _context.GetCompleteCode();
            uint32_t program = Runtime::ShaderCompiler::CompileComputeShader(shaderSource);
            glDeleteProgram(program);
            return true;
        } catch (const std::exception& e) {
            errorMessage = e.what();
            return false;
        }
    }

    InspectorKernel2D::InspectorKernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)>& Func,
                                         int WorkSizeX, int WorkSizeY) 
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

    bool InspectorKernel2D::Compile() {
        std::string unused;
        return Compile(unused);
    }

    bool InspectorKernel2D::Compile(std::string& errorMessage) {
        try {
            Runtime::AutoInitContext();
            Runtime::ContextGuard guard(Runtime::Context::GetInstance());
            
            std::string shaderSource = _context.GetCompleteCode();
            uint32_t program = Runtime::ShaderCompiler::CompileComputeShader(shaderSource);
            glDeleteProgram(program);
            return true;
        } catch (const std::exception& e) {
            errorMessage = e.what();
            return false;
        }
    }

    InspectorKernel3D::InspectorKernel3D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)>& Func,
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

    bool InspectorKernel3D::Compile() {
        std::string unused;
        return Compile(unused);
    }

    bool InspectorKernel3D::Compile(std::string& errorMessage) {
        try {
            Runtime::AutoInitContext();
            Runtime::ContextGuard guard(Runtime::Context::GetInstance());
            
            std::string shaderSource = _context.GetCompleteCode();
            uint32_t program = Runtime::ShaderCompiler::CompileComputeShader(shaderSource);
            glDeleteProgram(program);
            return true;
        } catch (const std::exception& e) {
            errorMessage = e.what();
            return false;
        }
    }

    // ===================================================================================
    // Executable Kernels
    // ===================================================================================

    Kernel1D::Kernel1D(const std::function<void(IR::Value::Var<int> &Id)>& Func, int WorkSizeX) 
        : _context(1), _name("Kernel1D") {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;

        IR::Value::Var<int> Id("(int(gl_GlobalInvocationID.x))");
        Func(Id);
    }

    Kernel1D::Kernel1D(const std::string& name, const std::function<void(IR::Value::Var<int> &Id)>& Func, int WorkSizeX) 
        : _context(1), _name(name) {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;

        IR::Value::Var<int> Id("(int(gl_GlobalInvocationID.x))");
        Func(Id);
    }

    void Kernel1D::SetName(const std::string& name) {
        _name = name;
    }

    std::string Kernel1D::GetName() const {
        return _name;
    }

    void Kernel1D::Dispatch(int GroupX, bool sync) {
        auto& profiler = KernelProfiler::GetInstance();
        unsigned int queryId = profiler.BeginQuery();
        ExecuteComputeDispatch(_context, GroupX, 1, 1, sync);
        profiler.EndQuery(queryId, _name, GroupX, 1, 1);
    }

    std::string Kernel1D::GetCode() {
        return _context.GetCompleteCode();
    }

    Kernel2D::Kernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)>& Func,
                       int WorkSizeX, int WorkSizeY) : _context(2), _name("Kernel2D") {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;
        _context.WorkSizeY = WorkSizeY;

        IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
        IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
        Func(IdX, IdY);
    }

    Kernel2D::Kernel2D(const std::string& name, const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)>& Func,
                       int WorkSizeX, int WorkSizeY) : _context(2), _name(name) {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;
        _context.WorkSizeY = WorkSizeY;

        IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
        IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
        Func(IdX, IdY);
    }

    void Kernel2D::SetName(const std::string& name) {
        _name = name;
    }

    std::string Kernel2D::GetName() const {
        return _name;
    }

    void Kernel2D::Dispatch(int GroupX, int GroupY, bool sync) {
        auto& profiler = KernelProfiler::GetInstance();
        unsigned int queryId = profiler.BeginQuery();
        ExecuteComputeDispatch(_context, GroupX, GroupY, 1, sync);
        profiler.EndQuery(queryId, _name, GroupX, GroupY, 1);
    }

    std::string Kernel2D::GetCode() {
        return _context.GetCompleteCode();
    }

    Kernel3D::Kernel3D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)>& Func,
                       int WorkSizeX, int WorkSizeY, int WorkSizeZ) : _context(3), _name("Kernel3D") {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;
        _context.WorkSizeY = WorkSizeY;
        _context.WorkSizeZ = WorkSizeZ;

        IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
        IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
        IR::Value::Var<int> IdZ("(int(gl_GlobalInvocationID.z))");
        Func(IdX, IdY, IdZ);
    }

    Kernel3D::Kernel3D(const std::string& name, const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)>& Func,
                       int WorkSizeX, int WorkSizeY, int WorkSizeZ) : _context(3), _name(name) {
        KernelBuilderGuard guard(IR::Builder::Builder::Get(), _context);

        _context.WorkSizeX = WorkSizeX;
        _context.WorkSizeY = WorkSizeY;
        _context.WorkSizeZ = WorkSizeZ;

        IR::Value::Var<int> IdX("(int(gl_GlobalInvocationID.x))");
        IR::Value::Var<int> IdY("(int(gl_GlobalInvocationID.y))");
        IR::Value::Var<int> IdZ("(int(gl_GlobalInvocationID.z))");
        Func(IdX, IdY, IdZ);
    }

    void Kernel3D::SetName(const std::string& name) {
        _name = name;
    }

    std::string Kernel3D::GetName() const {
        return _name;
    }

    void Kernel3D::Dispatch(int GroupX, int GroupY, int GroupZ, bool sync) {
        auto& profiler = KernelProfiler::GetInstance();
        unsigned int queryId = profiler.BeginQuery();
        ExecuteComputeDispatch(_context, GroupX, GroupY, GroupZ, sync);
        profiler.EndQuery(queryId, _name, GroupX, GroupY, GroupZ);
    }

    std::string Kernel3D::GetCode() {
        return _context.GetCompleteCode();
    }
}
