#pragma once

/**
 * Kernel.h:
 *      @Descripiton    :   The kernel function definition
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_KERNEL_H
#define EASYGPU_KERNEL_H

#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>

#include <IR/Value/Var.h>
#include <IR/Value/VarArray.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <functional>
#include <string>

namespace GPU::Kernel {
    // Forward declaration
    class KernelProfiler;

    /**
     * Base class for all kernels providing common functionality
     */
    class KernelBase {
    public:
        virtual ~KernelBase() = default;

    public:
        /**
         * Insert a memory barrier in the kernel code (GLSL barrier() for shader synchronization)
         * This is for shader-internal workgroup synchronization
         */
        static void WorkgroupBarrier();

        /**
         * Insert a memory barrier for GPU memory operations
         * This ensures all memory writes are visible to subsequent operations
         */
        static void MemoryBarrier();

        /**
         * Combined barrier: memory barrier + execution barrier
         * Ensures all threads in the workgroup reach this point and memory is synchronized
         */
        static void FullBarrier();

    public:
        /**
         * Runtime barrier after dispatch - ensures GPU execution is complete
         * This is called automatically if sync=true in Dispatch
         */
        static void RuntimeBarrier();
    };

    // ===================================================================================
    // Inspector Kernels - For debugging and viewing generated GLSL code
    // ===================================================================================

    /**
     * Inspector kernel for 1D workload - prints generated GLSL code instead of executing
     */
    class InspectorKernel1D : public KernelBase {
    public:
        InspectorKernel1D(const std::function<void(IR::Value::Var<int>& Id)>& Func, int WorkSizeX = 256);

    public:
        /**
         * Print the generated GLSL code to stdout
         */
        void PrintCode();

        /**
         * Get the generated GLSL code as string
         */
        std::string GetCode();

        /**
         * Compile the kernel to verify GLSL code is valid
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile();

        /**
         * Compile and get error message if failed
         * @param[out] errorMessage Compilation error message if failed
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile(std::string& errorMessage);

    private:
        KernelBuildContext _context;
    };

    /**
     * Backward compatibility alias for InspectorKernel1D
     * @deprecated Use InspectorKernel1D instead
     */
    using InspectorKernel = InspectorKernel1D;

    /**
     * Inspector kernel for 2D workload - prints generated GLSL code instead of executing
     */
    class InspectorKernel2D : public KernelBase {
    public:
        InspectorKernel2D(const std::function<void(IR::Value::Var<int>& IdX, IR::Value::Var<int>& IdY)>& Func,
                          int WorkSizeX = 16, int WorkSizeY = 16);

    public:
        /**
         * Print the generated GLSL code to stdout
         */
        void PrintCode();

        /**
         * Get the generated GLSL code as string
         */
        std::string GetCode();

        /**
         * Compile the kernel to verify GLSL code is valid
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile();

        /**
         * Compile and get error message if failed
         * @param[out] errorMessage Compilation error message if failed
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile(std::string& errorMessage);

    private:
        KernelBuildContext _context;
    };

    /**
     * Inspector kernel for 3D workload - prints generated GLSL code instead of executing
     */
    class InspectorKernel3D : public KernelBase {
    public:
        InspectorKernel3D(const std::function<void(IR::Value::Var<int>& IdX, IR::Value::Var<int>& IdY, IR::Value::Var<int>& IdZ)>& Func,
                          int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

    public:
        /**
         * Print the generated GLSL code to stdout
         */
        void PrintCode();

        /**
         * Get the generated GLSL code as string
         */
        std::string GetCode();

        /**
         * Compile the kernel to verify GLSL code is valid
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile();

        /**
         * Compile and get error message if failed
         * @param[out] errorMessage Compilation error message if failed
         * @return true if compilation succeeded, false otherwise
         */
        bool Compile(std::string& errorMessage);

    private:
        KernelBuildContext _context;
    };

    // ===================================================================================
    // Executable Kernels
    // ===================================================================================

    /**
     * The kernel class, which is the main API for the users to interact with GPU, provides
     * the way to construct a kernel function in graphics API. This kind of kernel is designed
     * for the 1 dimension work load
     */
    class Kernel1D : public KernelBase {
    public:
        /**
         * The 1 dimension kernel
         * @param Func The embedded DSL function
         * @param WorkSizeX The work size of x dimension (default 256)
         */
        Kernel1D(const std::function<void(IR::Value::Var<int> &Id)>& Func, int WorkSizeX = 256);

        /**
         * The 1 dimension kernel with name
         * @param name The kernel name for profiling identification
         * @param Func The embedded DSL function
         * @param WorkSizeX The work size of x dimension (default 256)
         */
        Kernel1D(const std::string& name, const std::function<void(IR::Value::Var<int> &Id)>& Func, int WorkSizeX = 256);

    public:
        /**
         * Set the kernel name for profiling
         * @param name The kernel name
         */
        void SetName(const std::string& name);

        /**
         * Get the kernel name
         * @return The kernel name
         */
        std::string GetName() const;

        /**
         * Dispatching the compute shader
         * Automatically binds all buffers that were bound via Bind() in the kernel function
         * @param GroupX The x group size
         * @param sync If true, wait for GPU execution to complete (blocking)
         */
        void Dispatch(int GroupX, bool sync = false);

        /**
         * Get the generated GLSL code without executing
         */
        std::string GetCode();

    private:
        KernelBuildContext _context;
        std::string _name = "Kernel1D";
    };

    /**
     * The kernel class for 2 dimension work load
     */
    class Kernel2D : public KernelBase {
    public:
        /**
         * The 2 dimension kernel
         * @param Func The embedded DSL function, receives (IdX, IdY)
         * @param WorkSizeX The work size of x dimension (default 16)
         * @param WorkSizeY The work size of y dimension (default 16)
         */
        Kernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)>& Func, 
                 int WorkSizeX = 16, int WorkSizeY = 16);

        /**
         * The 2 dimension kernel with name
         * @param name The kernel name for profiling identification
         * @param Func The embedded DSL function, receives (IdX, IdY)
         * @param WorkSizeX The work size of x dimension (default 16)
         * @param WorkSizeY The work size of y dimension (default 16)
         */
        Kernel2D(const std::string& name, const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)>& Func, 
                 int WorkSizeX = 16, int WorkSizeY = 16);

    public:
        /**
         * Set the kernel name for profiling
         * @param name The kernel name
         */
        void SetName(const std::string& name);

        /**
         * Get the kernel name
         * @return The kernel name
         */
        std::string GetName() const;

        /**
         * Dispatching the compute shader
         * @param GroupX The x group count
         * @param GroupY The y group count
         * @param sync If true, wait for GPU execution to complete (blocking)
         */
        void Dispatch(int GroupX, int GroupY, bool sync = false);

        /**
         * Get the generated GLSL code without executing
         */
        std::string GetCode();

    private:
        KernelBuildContext _context;
        std::string _name = "Kernel2D";
    };

    /**
     * The kernel class for 3 dimension work load
     */
    class Kernel3D : public KernelBase {
    public:
        /**
         * The 3 dimension kernel
         * @param Func The embedded DSL function, receives (IdX, IdY, IdZ)
         * @param WorkSizeX The work size of x dimension (default 8)
         * @param WorkSizeY The work size of y dimension (default 8)
         * @param WorkSizeZ The work size of z dimension (default 4)
         */
        Kernel3D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)>& Func, 
                 int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

        /**
         * The 3 dimension kernel with name
         * @param name The kernel name for profiling identification
         * @param Func The embedded DSL function, receives (IdX, IdY, IdZ)
         * @param WorkSizeX The work size of x dimension (default 8)
         * @param WorkSizeY The work size of y dimension (default 8)
         * @param WorkSizeZ The work size of z dimension (default 4)
         */
        Kernel3D(const std::string& name, const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)>& Func, 
                 int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

    public:
        /**
         * Set the kernel name for profiling
         * @param name The kernel name
         */
        void SetName(const std::string& name);

        /**
         * Get the kernel name
         * @return The kernel name
         */
        std::string GetName() const;

        /**
         * Dispatching the compute shader
         * @param GroupX The x group count
         * @param GroupY The y group count
         * @param GroupZ The z group count
         * @param sync If true, wait for GPU execution to complete (blocking)
         */
        void Dispatch(int GroupX, int GroupY, int GroupZ, bool sync = false);

        /**
         * Get the generated GLSL code without executing
         */
        std::string GetCode();

    private:
        KernelBuildContext _context;
        std::string _name = "Kernel3D";
    };
}

#endif //EASYGPU_KERNEL_H
