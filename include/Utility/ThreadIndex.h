#pragma once

/**
 * ThreadIndex.h:
 *      @Descripiton    :   Thread index utilities for GPU kernels
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */
#ifndef EASYGPU_THREADINDEX_H
#define EASYGPU_THREADINDEX_H

#include <IR/Value/Var.h>

namespace GPU {

// ===================================================================================
// Local Thread ID (within workgroup)
// ===================================================================================

/**
 * Get local thread ID within the workgroup (X dimension)
 * @return Var<int> Local invocation ID (0 to workgroup_size_x - 1)
 */
[[nodiscard]] inline IR::Value::Var<int> LocalThreadIdX() {
	return IR::Value::Var<int>("(int(gl_LocalInvocationID.x))");
}

/**
 * Get local thread ID within the workgroup (Y dimension)
 * @return Var<int> Local invocation ID (0 to workgroup_size_y - 1)
 */
[[nodiscard]] inline IR::Value::Var<int> LocalThreadIdY() {
	return IR::Value::Var<int>("(int(gl_LocalInvocationID.y))");
}

/**
 * Get local thread ID within the workgroup (Z dimension)
 * @return Var<int> Local invocation ID (0 to workgroup_size_z - 1)
 */
[[nodiscard]] inline IR::Value::Var<int> LocalThreadIdZ() {
	return IR::Value::Var<int>("(int(gl_LocalInvocationID.z))");
}

/**
 * Get local thread ID within the workgroup (1D)
 * Alias for LocalThreadIdX()
 * @return Var<int> Local invocation ID
 */
[[nodiscard]] inline IR::Value::Var<int> LocalThreadId() {
	return LocalThreadIdX();
}

// ===================================================================================
// Workgroup ID
// ===================================================================================

/**
 * Get workgroup ID (X dimension)
 * @return Var<int> Workgroup ID
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupIdX() {
	return IR::Value::Var<int>("(int(gl_WorkGroupID.x))");
}

/**
 * Get workgroup ID (Y dimension)
 * @return Var<int> Workgroup ID
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupIdY() {
	return IR::Value::Var<int>("(int(gl_WorkGroupID.y))");
}

/**
 * Get workgroup ID (Z dimension)
 * @return Var<int> Workgroup ID
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupIdZ() {
	return IR::Value::Var<int>("(int(gl_WorkGroupID.z))");
}

/**
 * Get workgroup ID (1D)
 * Alias for WorkgroupIdX()
 * @return Var<int> Workgroup ID
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupId() {
	return WorkgroupIdX();
}

// ===================================================================================
// Global Thread ID
// ===================================================================================

/**
 * Get global thread ID (X dimension)
 * @return Var<int> Global invocation ID
 */
[[nodiscard]] inline IR::Value::Var<int> GlobalThreadIdX() {
	return IR::Value::Var<int>("(int(gl_GlobalInvocationID.x))");
}

/**
 * Get global thread ID (Y dimension)
 * @return Var<int> Global invocation ID
 */
[[nodiscard]] inline IR::Value::Var<int> GlobalThreadIdY() {
	return IR::Value::Var<int>("(int(gl_GlobalInvocationID.y))");
}

/**
 * Get global thread ID (Z dimension)
 * @return Var<int> Global invocation ID
 */
[[nodiscard]] inline IR::Value::Var<int> GlobalThreadIdZ() {
	return IR::Value::Var<int>("(int(gl_GlobalInvocationID.z))");
}

// ===================================================================================
// Workgroup Size
// ===================================================================================

/**
 * Get workgroup size (X dimension)
 * @return Var<int> Workgroup size
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupSizeX() {
	return IR::Value::Var<int>("(int(gl_WorkGroupSize.x))");
}

/**
 * Get workgroup size (Y dimension)
 * @return Var<int> Workgroup size
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupSizeY() {
	return IR::Value::Var<int>("(int(gl_WorkGroupSize.y))");
}

/**
 * Get workgroup size (Z dimension)
 * @return Var<int> Workgroup size
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupSizeZ() {
	return IR::Value::Var<int>("(int(gl_WorkGroupSize.z))");
}

/**
 * Get workgroup size (1D)
 * Alias for WorkgroupSizeX()
 * @return Var<int> Workgroup size
 */
[[nodiscard]] inline IR::Value::Var<int> WorkgroupSize() {
	return WorkgroupSizeX();
}

// ===================================================================================
// Helper Structs for 2D/3D Access
// ===================================================================================

/**
 * 2D Local thread ID helper
 */
struct LocalId2DStruct {
	[[nodiscard]] IR::Value::Var<int> x() const {
		return LocalThreadIdX();
	}
	[[nodiscard]] IR::Value::Var<int> y() const {
		return LocalThreadIdY();
	}
};

/**
 * 2D Workgroup ID helper
 */
struct WorkgroupId2DStruct {
	[[nodiscard]] IR::Value::Var<int> x() const {
		return WorkgroupIdX();
	}
	[[nodiscard]] IR::Value::Var<int> y() const {
		return WorkgroupIdY();
	}
};

/**
 * 2D Global thread ID helper
 */
struct GlobalId2DStruct {
	[[nodiscard]] IR::Value::Var<int> x() const {
		return GlobalThreadIdX();
	}
	[[nodiscard]] IR::Value::Var<int> y() const {
		return GlobalThreadIdY();
	}
};

/**
 * Get 2D local thread ID
 * @return LocalId2DStruct with x() and y() methods
 */
[[nodiscard]] inline LocalId2DStruct LocalThreadId2D() {
	return LocalId2DStruct{};
}

/**
 * Get 2D workgroup ID
 * @return WorkgroupId2DStruct with x() and y() methods
 */
[[nodiscard]] inline WorkgroupId2DStruct WorkgroupId2D() {
	return WorkgroupId2DStruct{};
}

/**
 * Get 2D global thread ID
 * @return GlobalId2DStruct with x() and y() methods
 */
[[nodiscard]] inline GlobalId2DStruct GlobalThreadId2D() {
	return GlobalId2DStruct{};
}

} // namespace GPU

#endif // EASYGPU_THREADINDEX_H
