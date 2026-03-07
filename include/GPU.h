#pragma once

// X11 defines Bool as typedef int Bool, which conflicts with our Bool alias
// Must be done before any X11 headers are included
#ifdef Bool
#undef Bool
#endif

/**
 * @file GPU.h
 * @brief EasyGPU lazy header - includes everything with all namespaces
 *
 * This is a convenience header that includes all EasyGPU components
 * and brings all commonly used namespaces into scope.
 *
 * Usage:
 *   #include <GPU.h>
 *
 *   int main() {
 *       // All EasyGPU types are directly available
 *       Buffer<float> data(1024);
 *       Kernel::Kernel1D kernel(...);
 *       ...
 *   }
 */

// =============================================================================
// Core Kernel
// =============================================================================
#include <Kernel/FragmentKernel.h>
#include <Kernel/Kernel.h>
#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>

// =============================================================================
// Callable Functions
// =============================================================================
#include <Callable/Callable.h>

// =============================================================================
// IR Value Types - Order matters for template specializations
// =============================================================================
#include <IR/Value/BufferRef.h>
#include <IR/Value/TextureRef.h>
#include <IR/Value/Value.h>
// Include vector/matrix expression specializations BEFORE Expr.h
#include <IR/Value/Expr.h>
#include <IR/Value/ExprIVector.h>
#include <IR/Value/ExprMatrix.h>
#include <IR/Value/ExprVector.h>
// Var.h includes VarVector, VarIVector, VarMatrix at end
#include <IR/Value/Var.h>
#include <IR/Value/VarArray.h>
#include <IR/Value/VarStruct.h>

// =============================================================================
// IR Nodes (advanced usage)
// =============================================================================
#include <IR/Builder/Builder.h>
#include <IR/Builder/BuilderContext.h>
#include <IR/Node/Node.h>

// =============================================================================
// Control Flow
// =============================================================================
#include <Flow/BreakFlow.h>
#include <Flow/CodeCollectContext.h>
#include <Flow/ContinueFlow.h>
#include <Flow/DoWhileFlow.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/ReturnFlow.h>
#include <Flow/WhileFlow.h>

// =============================================================================
// Runtime
// =============================================================================
#include <Runtime/Buffer.h>
#include <Runtime/BufferSlot.h>
#include <Runtime/Context.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/ShaderException.h>
#include <Runtime/ShaderUtils.h>
#include <Runtime/Texture.h>
#include <Runtime/TextureSlot.h>
#include <Runtime/Uniform.h>

// =============================================================================
// Utilities
// =============================================================================
#include <Utility/Helpers.h>
#include <Utility/Math.h>
#include <Utility/Matrix.h>
#include <Utility/Meta/Std430Layout.h>
#include <Utility/Meta/StructMeta.h>
#include <Utility/Unref.h>
#include <Utility/Vec.h>

// =============================================================================
// Namespace Usings (Lazy Mode)
// =============================================================================

/// Root namespace for all EasyGPU components
using namespace GPU;

/// IR value types: Var<T>, Expr<T>, VarVector, Float3, etc.
using namespace GPU::IR::Value;

/// Runtime types: Buffer, Texture, Context
using namespace GPU::Runtime;

/// Math library
using namespace GPU::Math;

/// Callable function support
using namespace GPU::Callables;

/// Control flow: If, For, While, Break, Continue, Return
using namespace GPU::Flow;

/// Utility functions: Unref
using namespace GPU::Utility;

// =============================================================================
// Convenience Type Aliases
// =============================================================================

namespace GPU {

/// Alias for Kernel3D
using Kernel3D			= Kernel::Kernel3D;

/// Alias for Kernel2D
using Kernel2D			= Kernel::Kernel2D;

/// Alias for Kernel1D
using Kernel1D			= Kernel::Kernel1D;

/// Alias for InspectorKernel3D
using InspectorKernel1D = Kernel::InspectorKernel1D;

/// Alias for InspectorKernel2D
using InspectorKernel2D = Kernel::InspectorKernel2D;

/// Alias for InspectorKernel3D
using InspectorKernel3D = Kernel::InspectorKernel3D;

#ifdef _WIN32
using FragmentKernel2D = Kernel::FragmentKernel2D;
#endif
} // namespace GPU
