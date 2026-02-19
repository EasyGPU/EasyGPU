#pragma once

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

#pragma once

// =============================================================================
// Core Kernel
// =============================================================================
#include <Kernel/Kernel.h>
#include <Kernel/FragmentKernel.h>
#include <Kernel/KernelBuildContext.h>
#include <Kernel/KernelProfiler.h>

// =============================================================================
// Callable Functions
// =============================================================================
#include <Callable/Callable.h>

// =============================================================================
// IR Value Types
// =============================================================================
#include <IR/Value/Value.h>
#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Value/VarArray.h>
#include <IR/Value/VarVector.h>
#include <IR/Value/VarIVector.h>
#include <IR/Value/VarMatrix.h>
#include <IR/Value/VarStruct.h>
#include <IR/Value/ExprVector.h>
#include <IR/Value/ExprIVector.h>
#include <IR/Value/ExprMatrix.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/TextureRef.h>

// =============================================================================
// IR Nodes (advanced usage)
// =============================================================================
#include <IR/Node/Node.h>
#include <IR/Builder/Builder.h>
#include <IR/Builder/BuilderContext.h>

// =============================================================================
// Control Flow
// =============================================================================
#include <Flow/IfFlow.h>
#include <Flow/ForFlow.h>
#include <Flow/WhileFlow.h>
#include <Flow/DoWhileFlow.h>
#include <Flow/BreakFlow.h>
#include <Flow/ContinueFlow.h>
#include <Flow/ReturnFlow.h>
#include <Flow/CodeCollectContext.h>

// =============================================================================
// Runtime
// =============================================================================
#include <Runtime/Buffer.h>
#include <Runtime/Texture.h>
#include <Runtime/Uniform.h>
#include <Runtime/Context.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/ShaderException.h>
#include <Runtime/ShaderUtils.h>

// =============================================================================
// Utilities
// =============================================================================
#include <Utility/Vec.h>
#include <Utility/Matrix.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>
#include <Utility/Meta/StructMeta.h>
#include <Utility/Meta/Std430Layout.h>

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

// =============================================================================
// Convenience Type Aliases
// =============================================================================

namespace GPU {

/// Alias for Kernel3D
using Kernel3D = Kernel::Kernel3D;

/// Alias for Kernel2D
using Kernel2D = Kernel::Kernel2D;

/// Alias for Kernel1D
using Kernel1D = Kernel::Kernel1D;

/// Alias for InspectorKernel3D
using InspectorKernel1D = Kernel::InspectorKernel1D;

/// Alias for InspectorKernel2D
using InspectorKernel2D = Kernel::InspectorKernel2D;

/// Alias for InspectorKernel3D
using InspectorKernel3D = Kernel::InspectorKernel3D;

using FragmentKernel2D = Kernel::FragmentKernel2D;
} // namespace GPU
