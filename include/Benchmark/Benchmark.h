#pragma once

/**
 * @file Benchmark.h
 * @brief Benchmark suite system for measuring GPU kernel and operation performance.
 *
 * Provides a lightweight benchmarking framework that uses wall-clock time
 * (std::chrono) for portability across backends.  For GPU-side timer queries,
 * use Kernel::KernelProfiler directly.
 *
 * Typical usage:
 * @code
 *   GPU::Benchmark::BenchmarkSuite suite;
 *   suite.Add("vector_add", [&]() {
 *       kernel.Dispatch(64, true);
 *   });
 *   suite.Run();
 *   suite.PrintResults();
 * @endcode
 */

#ifndef EASYGPU_BENCHMARK_H
#define EASYGPU_BENCHMARK_H

#include <Runtime/Context.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace GPU::Benchmark {

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration for a single benchmark run.
 *
 * Controls warm-up and measurement iteration counts.  Warm-up dispatches are
 * executed but not recorded; they prime the GPU pipeline and shader cache so
 * that measured iterations reflect steady-state performance.
 */
struct BenchmarkConfig {
	/** Number of warm-up dispatches executed before measurement begins. */
	int	 warmupIterations   = 5;

	/** Number of measured dispatches whose timing is recorded. */
	int	 measuredIterations = 20;

	/** Issue a GPU-side finish (glFinish / vkQueueWaitIdle) after each dispatch
	 * for accurate per-dispatch timing rather than pipelined measurement. */
	bool syncAfterEach	   = true;

	/**
	 * @brief Construct a benchmark config with defaults.
	 * @param warmup  Warm-up iterations (default 5).
	 * @param measure Measured iterations (default 20).
	 * @param sync    Whether to synchronise after each dispatch (default true).
	 */
	BenchmarkConfig(int warmup = 5, int measure = 20, bool sync = true)
		: warmupIterations(warmup), measuredIterations(measure), syncAfterEach(sync) {
	}
};

// =============================================================================
// Result
// =============================================================================

/**
 * @brief Statistical result of a single benchmark.
 *
 * Contains aggregate metrics (min, max, avg, median, standard deviation)
 * computed from per-iteration wall-clock timings.
 */
struct BenchmarkResult {
	std::string			   name;			  ///< Human-readable benchmark name.
	int					   warmupCount		  = 0;  ///< Number of warm-up runs executed.
	int					   measuredCount	  = 0;  ///< Number of measured runs.
	double				   minMs			  = 0.0;  ///< Minimum iteration time in milliseconds.
	double				   maxMs			  = 0.0;  ///< Maximum iteration time in milliseconds.
	double				   avgMs			  = 0.0;  ///< Mean iteration time in milliseconds.
	double				   medianMs			  = 0.0;  ///< Median iteration time in milliseconds.
	double				   stddevMs			  = 0.0;  ///< Sample standard deviation in milliseconds.
	double				   totalMs			  = 0.0;  ///< Sum of all measured iteration times.
	std::vector<double>	   individualTimesMs; ///< Per-iteration timing in milliseconds.

	/**
	 * @brief Compute statistics from raw timing data.
	 * @param timesMs Vector of per-iteration times in milliseconds.
	 */
	void ComputeFromTimes(const std::vector<double> &timesMs) {
		individualTimesMs = timesMs;
		measuredCount	  = static_cast<int>(timesMs.size());

		if (timesMs.empty()) {
			return;
		}

		std::vector<double> sorted = timesMs;
		std::sort(sorted.begin(), sorted.end());

		minMs	= sorted.front();
		maxMs	= sorted.back();
		totalMs = std::accumulate(sorted.begin(), sorted.end(), 0.0);
		avgMs	= totalMs / static_cast<double>(sorted.size());

		// Median
		const size_t n = sorted.size();
		if (n % 2 == 1) {
			medianMs = sorted[n / 2];
		} else {
			medianMs = (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5;
		}

		// Sample standard deviation
		if (n > 1) {
			double variance = 0.0;
			for (double t : sorted) {
				const double diff = t - avgMs;
				variance += diff * diff;
			}
			variance /= static_cast<double>(n - 1);
			stddevMs = std::sqrt(variance);
		}
	}
};

// =============================================================================
// Runner
// =============================================================================

/**
 * @brief Standalone benchmark runner for ad-hoc timing of GPU operations.
 *
 * Uses wall-clock time (std::chrono) for portability across OpenGL and Vulkan
 * backends.  For GPU-side timer queries, use Kernel::KernelProfiler directly.
 *
 * @code
 *   BenchmarkRunner runner;
 *   runner.RunAndRecord("my_kernel", [&]() {
 *       kernel.Dispatch(64, true);
 *   });
 *   runner.PrintResults();
 * @endcode
 */
class BenchmarkRunner {
public:
	/**
	 * @brief Run a single named benchmark and store the result.
	 * @param name  Human-readable label for the benchmark.
	 * @param func  Callable that performs one GPU dispatch / operation.
	 * @param config  Iteration and synchronisation settings.
	 */
	void RunAndRecord(const std::string &name, std::function<void()> func,
					  const BenchmarkConfig &config = {}) {
		std::vector<double> timesMs;
		timesMs.reserve(config.measuredIterations);

		// Warm-up phase
		for (int i = 0; i < config.warmupIterations; ++i) {
			func();
		}

		// Flush GPU work before measurement (if sync is enabled)
		if (config.syncAfterEach) {
			Runtime::Context::GetBackend()->Finish();
		}

		// Measurement phase — uses wall-clock time for portability.
		// For GPU-side timer queries, use Kernel::KernelProfiler directly.
		for (int i = 0; i < config.measuredIterations; ++i) {
			const auto t0 = std::chrono::high_resolution_clock::now();

			func();

			if (config.syncAfterEach) {
				Runtime::Context::GetBackend()->Finish();
			}

			const auto t1 = std::chrono::high_resolution_clock::now();
			const double elapsedMs =
				static_cast<double>(
					std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
				1'000'000.0;
			timesMs.push_back(elapsedMs);
		}

		BenchmarkResult result;
		result.name		   = name;
		result.warmupCount = config.warmupIterations;
		result.ComputeFromTimes(timesMs);
		_results.push_back(std::move(result));
	}

	/**
	 * @brief Get all recorded benchmark results.
	 * @return Const reference to the results vector.
	 */
	[[nodiscard]] const std::vector<BenchmarkResult> &GetResults() const {
		return _results;
	}

	/**
	 * @brief Clear all recorded results.
	 */
	void Clear() {
		_results.clear();
	}

	/**
	 * @brief Print formatted benchmark results to stdout.
	 */
	void PrintResults() const;

	/**
	 * @brief Get formatted benchmark results as a string.
	 * @return Formatted string suitable for logging or file output.
	 */
	[[nodiscard]] std::string GetFormattedResults() const;

private:
	std::vector<BenchmarkResult> _results;
};

// =============================================================================
// Suite
// =============================================================================

/**
 * @brief Organised collection of benchmarks executed as a group.
 *
 * Benchmarks are registered once via Add() and then run together with Run().
 * Results are collected and can be printed or queried individually.
 *
 * @code
 *   BenchmarkSuite suite("MySuite");
 *   suite.Add("kernel_a", [&]() { kernelA.Dispatch(64, true); });
 *   suite.Add("kernel_b", [&]() { kernelB.Dispatch(64, true); });
 *   suite.Run(BenchmarkConfig(5, 50));
 *   suite.PrintResults();
 * @endcode
 */
class BenchmarkSuite {
public:
	/**
	 * @brief Construct an empty benchmark suite.
	 * @param name Human-readable suite name displayed in results.
	 */
	explicit BenchmarkSuite(std::string name = "BenchmarkSuite") : _name(std::move(name)) {
	}

	/**
	 * @brief Register a benchmark function.
	 *
	 * Must be called before Run().  The function should perform one complete
	 * GPU operation (typically a single Dispatch() call).
	 *
	 * @param name Human-readable benchmark name.
	 * @param func Callable that performs one operation iteration.
	 */
	void Add(const std::string &name, std::function<void()> func) {
		_entries.push_back({name, std::move(func)});
	}

	/**
	 * @brief Run all registered benchmarks in registration order.
	 * @param config Iteration and synchronisation settings applied to every benchmark.
	 */
	void Run(const BenchmarkConfig &config = {});

	/**
	 * @brief Get all benchmark results (available after Run()).
	 * @return Const reference to the results vector.
	 */
	[[nodiscard]] const std::vector<BenchmarkResult> &GetResults() const {
		return _results;
	}

	/**
	 * @brief Clear all registered benchmarks and results.
	 */
	void Clear() {
		_entries.clear();
		_results.clear();
	}

	/**
	 * @brief Print formatted suite results to stdout.
	 */
	void PrintResults() const;

	/**
	 * @brief Get formatted suite results as a string.
	 * @return Formatted string suitable for logging or file output.
	 */
	[[nodiscard]] std::string GetFormattedResults() const;

private:
	struct Entry {
		std::string			   name;
		std::function<void()>  func;
	};

	std::string				   _name;
	std::vector<Entry>		   _entries;
	std::vector<BenchmarkResult> _results;
};

} // namespace GPU::Benchmark

#endif // EASYGPU_BENCHMARK_H
