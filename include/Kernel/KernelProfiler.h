#pragma once

/**
 * @file KernelProfiler.h
 * @brief Kernel profiling tool for measuring GPU execution time.
 */

#ifndef EASYGPU_KERNEL_PROFILER_H
#define EASYGPU_KERNEL_PROFILER_H

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace GPU::Kernel {

// Forward declaration
class KernelBuildContext;

/**
 * Query result for a specific kernel's profiling data
 */
struct KernelProfilerQueryResult {
	std::string kernelName;		   // Name of the kernel
	int			counter		= 0;   // Number of executions
	double		minTimeMs	= 0.0; // Minimum execution time in milliseconds
	double		maxTimeMs	= 0.0; // Maximum execution time in milliseconds
	double		avgTimeMs	= 0.0; // Average execution time in milliseconds
	double		totalTimeMs = 0.0; // Total execution time in milliseconds
};

/**
 * Single profiling record for one kernel execution
 */
struct KernelProfileRecord {
	std::string							  kernelName;			  // Name of the kernel
	double								  elapsedTimeMs;		  // Execution time in milliseconds
	int									  groupX, groupY, groupZ; // Dispatch dimensions
	std::chrono::system_clock::time_point timestamp;			  // When it was executed
};

/**
 * Kernel Profiler for measuring GPU compute shader execution time
 *
 * Usage:
 *   // Enable profiling
 *   KernelProfiler::GetInstance().SetEnabled(true);
 *
 *   // Run kernels...
 *   kernel.Dispatch(32, 32, true);  // sync=true required for accurate timing
 *
 *   // Print results
 *   KernelProfiler::GetInstance().PrintInfo("count");
 *
 *   // Or query specific kernel
 *   auto result = KernelProfiler::GetInstance().QueryInfo("MyKernel");
 *   std::cout << "Avg time: " << result.avgTimeMs << " ms\n";
 */
class KernelProfiler {
public:
	/**
	 * @brief Get the singleton instance.
	 * @return Reference to the KernelProfiler singleton.
	 */
	static KernelProfiler &GetInstance();

	// Disable copy and move
	KernelProfiler(const KernelProfiler &)			  = delete;
	KernelProfiler &operator=(const KernelProfiler &) = delete;
	KernelProfiler(KernelProfiler &&)				  = delete;
	KernelProfiler &operator=(KernelProfiler &&)	  = delete;

public:
	/**
	 * @brief Enable or disable profiling.
	 *
	 * When disabled, no records are collected.
	 * @param enabled True to enable profiling.
	 */
	void		 SetEnabled(bool enabled);

	/**
	 * @brief Check if profiling is enabled.
	 * @return true if profiling is active.
	 */
	bool		 IsEnabled() const;

	/**
	 * @brief Clear all profiling records and statistics.
	 */
	void		 Clear();

	/**
	 * @brief Begin profiling a kernel dispatch.
	 * @return Query ID for ending the timer, 0 if profiling is disabled.
	 */
	unsigned int BeginQuery();

	/**
	 * @brief End profiling a kernel dispatch and record the result.
	 * @param queryId The query ID from BeginQuery.
	 * @param kernelName Name of the kernel.
	 * @param groupX X dimension dispatch size.
	 * @param groupY Y dimension dispatch size.
	 * @param groupZ Z dimension dispatch size.
	 */
	void		 EndQuery(unsigned int queryId, const std::string &kernelName, int groupX, int groupY, int groupZ);

	/**
	 * @brief Begin profiling on the current OpenGL context (without context switch).
	 *
	 * Use this for FragmentKernel profiling where the context is already set.
	 * @return Query ID for ending the timer, 0 if profiling is disabled.
	 */
	unsigned int BeginQueryOnCurrentContext();

	/**
	 * @brief End profiling on the current OpenGL context (without context switch).
	 *
	 * Use this for FragmentKernel profiling where the context is already set.
	 * @param queryId The query ID from BeginQueryOnCurrentContext.
	 * @param kernelName Name of the kernel.
	 * @param groupX X dimension dispatch size.
	 * @param groupY Y dimension dispatch size.
	 * @param groupZ Z dimension dispatch size.
	 */
	void		 EndQueryOnCurrentContext(unsigned int queryId, const std::string &kernelName, int groupX, int groupY,
										  int groupZ);

public:
	/**
	 * @brief Query profiling statistics for a specific kernel by name.
	 * @param kernelName The name of the kernel to query.
	 * @return Query result with counter, min, max, avg times.
	 */
	KernelProfilerQueryResult				QueryInfo(const std::string &kernelName) const;

	/**
	 * @brief Get total elapsed time of all kernels recorded.
	 * @return Total time in milliseconds.
	 */
	double									GetTotalTime() const;

	/**
	 * @brief Print profiling results to stdout.
	 * @param mode "count" to print statistics (default), "trace" to print individual execution records.
	 */
	void									PrintInfo(const std::string &mode = "count") const;

	/**
	 * @brief Get formatted profiling results as string.
	 * @param mode "count" for statistics (default), "trace" for execution records.
	 * @return Formatted string with profiling results.
	 */
	std::string								GetFormattedOutput(const std::string &mode = "count") const;

	/**
	 * @brief Get all profiling records (trace mode).
	 * @return Vector of KernelProfileRecord entries.
	 */
	const std::vector<KernelProfileRecord> &GetRecords() const;

	/**
	 * @brief Get all kernel statistics.
	 * @return Vector of KernelProfilerQueryResult for all profiled kernels.
	 */
	std::vector<KernelProfilerQueryResult>	GetAllStats() const;

private:
	KernelProfiler() = default;
	~KernelProfiler();

	void		 InitializeQueries();
	void		 CleanupQueries();
	unsigned int AcquireQuery();
	void		 ReleaseQuery(unsigned int query);
	void		 RecordExecution(const std::string &kernelName, int groupX, int groupY, int groupZ, double elapsedMs);

private:
	mutable std::recursive_mutex											_mutex;

	bool																	_enabled = false;

	// Query pool for timer queries
	std::vector<unsigned int>												_queryPool;
	std::vector<unsigned int>												_availableQueries;
	static constexpr size_t													MAX_QUERIES = 64;

	// Profiling records (trace)
	std::vector<KernelProfileRecord>										_records;

	// Aggregated statistics per kernel name
	std::unordered_map<std::string, KernelProfilerQueryResult>				_stats;

	// Track which thread started each query to prevent cross-thread use
	std::unordered_map<unsigned int, std::thread::id>						_queryOwners;
	std::unordered_map<unsigned int, std::chrono::steady_clock::time_point> _cpuQueryStarts;
	unsigned int															_nextCpuQuery = 0x80000000u;
};

// ===================================================================================
// Helper macros and inline functions
// ===================================================================================

/**
 * RAII helper for automatic kernel profiling
 */
class KernelProfileScope {
public:
	/**
	 * @brief Begin a profiling scope for a kernel dispatch.
	 * @param kernelName Name of the kernel.
	 * @param groupX X dimension dispatch size.
	 * @param groupY Y dimension dispatch size (default 1).
	 * @param groupZ Z dimension dispatch size (default 1).
	 */
	KernelProfileScope(const std::string &kernelName, int groupX, int groupY = 1, int groupZ = 1);

	/**
	 * @brief End the profiling scope and record the result.
	 */
	~KernelProfileScope();

	// Disable copy and move
	KernelProfileScope(const KernelProfileScope &)			  = delete;
	KernelProfileScope &operator=(const KernelProfileScope &) = delete;
	KernelProfileScope(KernelProfileScope &&)				  = delete;
	KernelProfileScope &operator=(KernelProfileScope &&)	  = delete;

private:
	std::string	 _kernelName;
	int			 _groupX, _groupY, _groupZ;
	unsigned int _queryId;
};

/**
 * @brief Convenience functions for global profiler access.
 */

/**
 * @brief Enable or disable the global kernel profiler.
 * @param enabled True to enable (default).
 */
inline void EnableKernelProfiler(bool enabled = true) {
	KernelProfiler::GetInstance().SetEnabled(enabled);
}

/**
 * @brief Clear all global profiler records.
 */
inline void ClearKernelProfilerInfo() {
	KernelProfiler::GetInstance().Clear();
}

/**
 * @brief Print global profiler results to stdout.
 * @param mode "count" for statistics (default), "trace" for execution records.
 */
inline void PrintKernelProfilerInfo(const std::string &mode = "count") {
	KernelProfiler::GetInstance().PrintInfo(mode);
}

/**
 * @brief Query profiler statistics for a named kernel.
 * @param kernelName The kernel name.
 * @return Query result with timing statistics.
 */
inline KernelProfilerQueryResult QueryKernelProfilerInfo(const std::string &kernelName) {
	return KernelProfiler::GetInstance().QueryInfo(kernelName);
}

/**
 * @brief Get total elapsed time across all profiled kernels.
 * @return Total time in milliseconds.
 */
inline double GetKernelProfilerTotalTime() {
	return KernelProfiler::GetInstance().GetTotalTime();
}

/**
 * @brief Get formatted profiler output as string.
 * @param mode "count" for statistics (default), "trace" for execution records.
 * @return Formatted string with profiling results.
 */
inline std::string GetKernelProfilerFormattedOutput(const std::string &mode = "count") {
	return KernelProfiler::GetInstance().GetFormattedOutput(mode);
}

} // namespace GPU::Kernel

#endif // EASYGPU_KERNEL_PROFILER_H
