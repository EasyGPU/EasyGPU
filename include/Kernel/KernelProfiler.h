#pragma once

/**
 * KernelProfiler.h:
 *      @Descripiton    :   Kernel profiling tool for measuring GPU execution time
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 * 
 *  Reference: Taichi Kernel Profiler API
 *  https://docs.taichi-lang.org/api/taichi/profiler/kernel_profiler/
 */
#ifndef EASYGPU_KERNEL_PROFILER_H
#define EASYGPU_KERNEL_PROFILER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <memory>

namespace GPU::Kernel {

    // Forward declaration
    class KernelBuildContext;

    /**
     * Query result for a specific kernel's profiling data
     */
    struct KernelProfilerQueryResult {
        std::string kernelName;     // Name of the kernel
        int counter = 0;            // Number of executions
        double minTimeMs = 0.0;     // Minimum execution time in milliseconds
        double maxTimeMs = 0.0;     // Maximum execution time in milliseconds
        double avgTimeMs = 0.0;     // Average execution time in milliseconds
        double totalTimeMs = 0.0;   // Total execution time in milliseconds
    };

    /**
     * Single profiling record for one kernel execution
     */
    struct KernelProfileRecord {
        std::string kernelName;     // Name of the kernel
        double elapsedTimeMs;       // Execution time in milliseconds
        int groupX, groupY, groupZ; // Dispatch dimensions
        std::chrono::system_clock::time_point timestamp; // When it was executed
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
         * Get the singleton instance
         */
        static KernelProfiler& GetInstance();

        // Disable copy and move
        KernelProfiler(const KernelProfiler&) = delete;
        KernelProfiler& operator=(const KernelProfiler&) = delete;
        KernelProfiler(KernelProfiler&&) = delete;
        KernelProfiler& operator=(KernelProfiler&&) = delete;

    public:
        /**
         * Enable or disable profiling
         * When disabled, no records are collected
         */
        void SetEnabled(bool enabled);

        /**
         * Check if profiling is enabled
         */
        bool IsEnabled() const;

        /**
         * Clear all profiling records and statistics
         */
        void Clear();

        /**
         * Begin profiling a kernel dispatch
         * @return Query ID for ending the timer, 0 if profiling is disabled
         */
        unsigned int BeginQuery();

        /**
         * End profiling a kernel dispatch and record the result
         * @param queryId The query ID from BeginQuery
         * @param kernelName Name of the kernel
         * @param groupX X dimension dispatch size
         * @param groupY Y dimension dispatch size  
         * @param groupZ Z dimension dispatch size
         */
        void EndQuery(unsigned int queryId, const std::string& kernelName, 
                      int groupX, int groupY, int groupZ);

    public:
        /**
         * Query profiling statistics for a specific kernel by name
         * @param kernelName The name of the kernel to query
         * @return Query result with counter, min, max, avg times
         */
        KernelProfilerQueryResult QueryInfo(const std::string& kernelName) const;

        /**
         * Get total elapsed time of all kernels recorded
         * @return Total time in milliseconds
         */
        double GetTotalTime() const;

        /**
         * Print profiling results
         * @param mode "count" - print statistics (default)
         *             "trace" - print individual execution records
         */
        void PrintInfo(const std::string& mode = "count") const;

        /**
         * Get formatted profiling results as string
         * @param mode "count" - statistics (default), "trace" - execution records
         * @return Formatted string with profiling results
         */
        std::string GetFormattedOutput(const std::string& mode = "count") const;

        /**
         * Get all profiling records (trace mode)
         */
        const std::vector<KernelProfileRecord>& GetRecords() const;

        /**
         * Get all kernel statistics
         */
        std::vector<KernelProfilerQueryResult> GetAllStats() const;

    private:
        KernelProfiler() = default;
        ~KernelProfiler();

        void InitializeQueries();
        void CleanupQueries();
        unsigned int AcquireQuery();
        void ReleaseQuery(unsigned int query);

    private:
        bool _enabled = false;
        
        // Query pool for timer queries
        std::vector<unsigned int> _queryPool;
        std::vector<unsigned int> _availableQueries;
        static constexpr size_t MAX_QUERIES = 64;

        // Profiling records (trace)
        std::vector<KernelProfileRecord> _records;

        // Aggregated statistics per kernel name
        std::unordered_map<std::string, KernelProfilerQueryResult> _stats;
    };

    // ===================================================================================
    // Helper macros and inline functions
    // ===================================================================================

    /**
     * RAII helper for automatic kernel profiling
     */
    class KernelProfileScope {
    public:
        KernelProfileScope(const std::string& kernelName, int groupX, int groupY = 1, int groupZ = 1);
        ~KernelProfileScope();

        // Disable copy and move
        KernelProfileScope(const KernelProfileScope&) = delete;
        KernelProfileScope& operator=(const KernelProfileScope&) = delete;
        KernelProfileScope(KernelProfileScope&&) = delete;
        KernelProfileScope& operator=(KernelProfileScope&&) = delete;

    private:
        std::string _kernelName;
        int _groupX, _groupY, _groupZ;
        unsigned int _queryId;
    };

    /**
     * Convenience functions for global profiler access
     */
    inline void EnableKernelProfiler(bool enabled = true) {
        KernelProfiler::GetInstance().SetEnabled(enabled);
    }

    inline void ClearKernelProfilerInfo() {
        KernelProfiler::GetInstance().Clear();
    }

    inline void PrintKernelProfilerInfo(const std::string& mode = "count") {
        KernelProfiler::GetInstance().PrintInfo(mode);
    }

    inline KernelProfilerQueryResult QueryKernelProfilerInfo(const std::string& kernelName) {
        return KernelProfiler::GetInstance().QueryInfo(kernelName);
    }

    inline double GetKernelProfilerTotalTime() {
        return KernelProfiler::GetInstance().GetTotalTime();
    }

    inline std::string GetKernelProfilerFormattedOutput(const std::string& mode = "count") {
        return KernelProfiler::GetInstance().GetFormattedOutput(mode);
    }

}

#endif //EASYGPU_KERNEL_PROFILER_H
