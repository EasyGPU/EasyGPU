#pragma once

/**
 * @file Kernel.h
 * @brief The kernel function definition.
 */

#ifndef EASYGPU_KERNEL_H
#define EASYGPU_KERNEL_H

#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>

#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <IR/Value/VarArray.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <functional>
#include <string>

namespace GPU::Kernel {
// Forward declaration
class KernelProfiler;

/**
 * @brief Base class for all kernels providing common synchronization primitives.
 */
class KernelBase {
public:
	virtual ~KernelBase() = default;

public:
	/**
	 * @brief Insert a workgroup memory barrier (GLSL barrier()).
	 *
	 * This is for shader-internal workgroup synchronization.
	 */
	static void WorkgroupBarrier();

	/**
	 * @brief Insert a GPU memory barrier.
	 *
	 * Ensures all memory writes are visible to subsequent operations.
	 */
	static void MemoryBarrier();

	/**
	 * @brief Combined memory barrier and execution barrier.
	 *
	 * Ensures all threads in the workgroup reach this point and memory is synchronized.
	 */
	static void FullBarrier();

public:
	/**
	 * @brief Issue a runtime GPU barrier after dispatch.
	 *
	 * Ensures GPU execution is complete. Called automatically if sync=true in Dispatch.
	 */
	static void RuntimeBarrier();
};

// ===================================================================================
// Inspector Kernels - For debugging and viewing generated GLSL code
// ===================================================================================

/**
 * @brief Inspector kernel for 1D workload - prints generated GLSL code instead of executing.
 */
class InspectorKernel1D : public KernelBase {
public:
	/**
	 * @brief Construct an inspector kernel for 1D workload.
	 * @param Func The embedded DSL function receiving the thread index.
	 * @param WorkSizeX The work group size in the X dimension (default 256).
	 */
	InspectorKernel1D(const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX = 256);

public:
	/**
	 * @brief Print the generated GLSL code to stdout.
	 */
	void		PrintCode();

	/**
	 * @brief Get the generated GLSL code as string.
	 * @return The full GLSL source code.
	 */
	std::string GetCode();

	/**
	 * @brief Compile the kernel to verify GLSL code is valid.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile();

	/**
	 * @brief Compile and get error message if failed.
	 * @param[out] errorMessage Compilation error message if failed.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile(std::string &errorMessage);

private:
	KernelBuildContext _context;
};

/**
 * Backward compatibility alias for InspectorKernel1D
 * @deprecated Use InspectorKernel1D instead
 */
using InspectorKernel = InspectorKernel1D;

/**
 * @brief Inspector kernel for 2D workload - prints generated GLSL code instead of executing.
 */
class InspectorKernel2D : public KernelBase {
public:
	/**
	 * @brief Construct an inspector kernel for 2D workload.
	 * @param Func The embedded DSL function receiving (IdX, IdY) thread indices.
	 * @param WorkSizeX The work group size in the X dimension (default 16).
	 * @param WorkSizeY The work group size in the Y dimension (default 16).
	 */
	InspectorKernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func,
					  int WorkSizeX = 16, int WorkSizeY = 16);

public:
	/**
	 * @brief Print the generated GLSL code to stdout.
	 */
	void		PrintCode();

	/**
	 * @brief Get the generated GLSL code as string.
	 * @return The full GLSL source code.
	 */
	std::string GetCode();

	/**
	 * @brief Compile the kernel to verify GLSL code is valid.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile();

	/**
	 * @brief Compile and get error message if failed.
	 * @param[out] errorMessage Compilation error message if failed.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile(std::string &errorMessage);

private:
	KernelBuildContext _context;
};

/**
 * @brief Inspector kernel for 3D workload - prints generated GLSL code instead of executing.
 */
class InspectorKernel3D : public KernelBase {
public:
	/**
	 * @brief Construct an inspector kernel for 3D workload.
	 * @param Func The embedded DSL function receiving (IdX, IdY, IdZ) thread indices.
	 * @param WorkSizeX The work group size in the X dimension (default 8).
	 * @param WorkSizeY The work group size in the Y dimension (default 8).
	 * @param WorkSizeZ The work group size in the Z dimension (default 4).
	 */
	InspectorKernel3D(
		const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
		int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

public:
	/**
	 * @brief Print the generated GLSL code to stdout.
	 */
	void		PrintCode();

	/**
	 * @brief Get the generated GLSL code as string.
	 * @return The full GLSL source code.
	 */
	std::string GetCode();

	/**
	 * @brief Compile the kernel to verify GLSL code is valid.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile();

	/**
	 * @brief Compile and get error message if failed.
	 * @param[out] errorMessage Compilation error message if failed.
	 * @return true if compilation succeeded, false otherwise.
	 */
	bool		Compile(std::string &errorMessage);

private:
	KernelBuildContext _context;
};

// ===================================================================================
// Executable Kernels
// ===================================================================================

/**
 * @brief 1D compute kernel - the main API for single-dimension GPU workloads.
 *
 * Provides the way to construct a kernel function via the embedded DSL and dispatch
 * it as a compute shader.
 */
class Kernel1D : public KernelBase {
public:
	/**
	 * @brief Construct a 1D kernel.
	 * @param Func The embedded DSL function.
	 * @param WorkSizeX The work size of x dimension (default 256).
	 */
	Kernel1D(const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX = 256);

	/**
	 * @brief Construct a 1D kernel with a profiling name.
	 * @param name The kernel name for profiling identification.
	 * @param Func The embedded DSL function.
	 * @param WorkSizeX The work size of x dimension (default 256).
	 */
	Kernel1D(const std::string &name, const std::function<void(IR::Value::Var<int> &Id)> &Func, int WorkSizeX = 256);

public:
	/**
	 * @brief Set the kernel name for profiling.
	 * @param name The kernel name.
	 */
	void		SetName(const std::string &name);

	/**
	 * @brief Get the kernel name.
	 * @return The kernel name.
	 */
	std::string GetName() const;

	/**
	 * @brief Dispatch the compute shader.
	 *
	 * Automatically binds all buffers that were bound via Bind() in the kernel function.
	 * @param GroupX The x group size.
	 * @param sync If true, wait for GPU execution to complete (blocking).
	 */
	void		Dispatch(int GroupX, bool sync = false);

	/**
	 * @brief Get the generated GLSL code without executing.
	 * @return The full GLSL compute shader source.
	 */
	std::string GetCode();

	const KernelBuildContext &GetContext() const { return _context; }

private:
	KernelBuildContext _context;
	std::string		   _name = "Kernel1D";
};

/**
 * @brief 2D compute kernel for two-dimensional GPU workloads.
 */
class Kernel2D : public KernelBase {
public:
	/**
	 * @brief Construct a 2D kernel.
	 * @param Func The embedded DSL function, receives (IdX, IdY).
	 * @param WorkSizeX The work size of x dimension (default 16).
	 * @param WorkSizeY The work size of y dimension (default 16).
	 */
	Kernel2D(const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func, int WorkSizeX = 16,
			 int WorkSizeY = 16);

	/**
	 * @brief Construct a 2D kernel with a profiling name.
	 * @param name The kernel name for profiling identification.
	 * @param Func The embedded DSL function, receives (IdX, IdY).
	 * @param WorkSizeX The work size of x dimension (default 16).
	 * @param WorkSizeY The work size of y dimension (default 16).
	 */
	Kernel2D(const std::string															   &name,
			 const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY)> &Func, int WorkSizeX = 16,
			 int WorkSizeY = 16);

public:
	/**
	 * @brief Set the kernel name for profiling.
	 * @param name The kernel name.
	 */
	void		SetName(const std::string &name);

	/**
	 * @brief Get the kernel name.
	 * @return The kernel name.
	 */
	std::string GetName() const;

	/**
	 * @brief Dispatch the 2D compute shader.
	 * @param GroupX The x group count.
	 * @param GroupY The y group count.
	 * @param sync If true, wait for GPU execution to complete (blocking).
	 */
	void		Dispatch(int GroupX, int GroupY, bool sync = false);

	/**
	 * @brief Get the generated GLSL code without executing.
	 * @return The full GLSL compute shader source.
	 */
	std::string GetCode();

	const KernelBuildContext &GetContext() const { return _context; }

private:
	KernelBuildContext _context;
	std::string		   _name = "Kernel2D";
};

/**
 * @brief 3D compute kernel for three-dimensional GPU workloads.
 */
class Kernel3D : public KernelBase {
public:
	/**
	 * @brief Construct a 3D kernel.
	 * @param Func The embedded DSL function, receives (IdX, IdY, IdZ).
	 * @param WorkSizeX The work size of x dimension (default 8).
	 * @param WorkSizeY The work size of y dimension (default 8).
	 * @param WorkSizeZ The work size of z dimension (default 4).
	 */
	Kernel3D(
		const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
		int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

	/**
	 * @brief Construct a 3D kernel with a profiling name.
	 * @param name The kernel name for profiling identification.
	 * @param Func The embedded DSL function, receives (IdX, IdY, IdZ).
	 * @param WorkSizeX The work size of x dimension (default 8).
	 * @param WorkSizeY The work size of y dimension (default 8).
	 * @param WorkSizeZ The work size of z dimension (default 4).
	 */
	Kernel3D(
		const std::string																						&name,
		const std::function<void(IR::Value::Var<int> &IdX, IR::Value::Var<int> &IdY, IR::Value::Var<int> &IdZ)> &Func,
		int WorkSizeX = 8, int WorkSizeY = 8, int WorkSizeZ = 4);

public:
	/**
	 * @brief Set the kernel name for profiling.
	 * @param name The kernel name.
	 */
	void		SetName(const std::string &name);

	/**
	 * @brief Get the kernel name.
	 * @return The kernel name.
	 */
	std::string GetName() const;

	/**
	 * @brief Dispatch the 3D compute shader.
	 * @param GroupX The x group count.
	 * @param GroupY The y group count.
	 * @param GroupZ The z group count.
	 * @param sync If true, wait for GPU execution to complete (blocking).
	 */
	void		Dispatch(int GroupX, int GroupY, int GroupZ, bool sync = false);

	/**
	 * @brief Get the generated GLSL code without executing.
	 * @return The full GLSL compute shader source.
	 */
	std::string GetCode();

	const KernelBuildContext &GetContext() const { return _context; }

private:
	KernelBuildContext _context;
	std::string		   _name = "Kernel3D";
};
} // namespace GPU::Kernel

#endif // EASYGPU_KERNEL_H
