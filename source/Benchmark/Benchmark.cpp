/**
 * @file Benchmark.cpp
 * @brief Benchmark suite system implementation.
 */

#include <Benchmark/Benchmark.h>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace GPU::Benchmark {

// =============================================================================
// ANSI color helpers (internal)
// =============================================================================

namespace {
const char *COLOR_RESET	  = "\033[0m";
const char *COLOR_BOLD	  = "\033[1m";
const char *COLOR_CYAN	  = "\033[36m";
const char *COLOR_GREEN	  = "\033[32m";
const char *COLOR_YELLOW  = "\033[33m";
const char *COLOR_MAGENTA = "\033[35m";
const char *COLOR_GRAY	  = "\033[90m";

bool		UseColor() {
#ifdef _WIN32
	return false;
#else
	return true;
#endif
}

const char *Col(const char *color) {
	return UseColor() ? color : "";
}
} // namespace

// =============================================================================
// Formatting helpers (internal)
// =============================================================================

namespace {

void PrintHeader(std::ostream &os, bool useColor) {
	if (useColor) {
		os << Col(COLOR_CYAN)
		   << "╔═════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "╗\n"
		   << Col(COLOR_RESET);
		os << Col(COLOR_CYAN) << "║" << Col(COLOR_BOLD)
		   << "                    Benchmark Results                                     " << Col(COLOR_RESET)
		   << Col(COLOR_CYAN) << "║\n"
		   << Col(COLOR_RESET);
	} else {
		os << "+------------------------------------------------------------------------------+\n";
		os << "|                    Benchmark Results                                         |\n";
	}
}

void PrintSeparator(std::ostream &os, bool useColor) {
	if (useColor) {
		os << Col(COLOR_CYAN)
		   << "╠═════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "╣\n"
		   << Col(COLOR_RESET);
	} else {
		os << "+------------------------------------------------------------------------------+\n";
	}
}

void PrintFooter(std::ostream &os, bool useColor) {
	if (useColor) {
		os << Col(COLOR_CYAN)
		   << "╚═════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "══════════════════"
			  "╝\n"
		   << Col(COLOR_RESET);
	} else {
		os << "+------------------------------------------------------------------------------+\n";
	}
}

void PrintColumnHeaders(std::ostream &os, bool useColor) {
	if (useColor) {
		os << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET) << Col(COLOR_BOLD) << std::left << std::setw(24)
		   << "Benchmark" << Col(COLOR_RESET) << " │ " << std::right << std::setw(6) << "Count"
		   << " │ " << std::setw(9) << "Min(ms)"
		   << " │ " << std::setw(9) << "Avg(ms)"
		   << " │ " << std::setw(9) << "Median"
		   << " │ " << std::setw(9) << "Max(ms)"
		   << " │ " << std::setw(9) << "StdDev" << Col(COLOR_CYAN) << "  ║\n"
		   << Col(COLOR_RESET);
	} else {
		os << "| " << std::left << std::setw(24) << "Benchmark"
		   << " | " << std::right << std::setw(6) << "Count"
		   << " | " << std::setw(9) << "Min(ms)"
		   << " | " << std::setw(9) << "Avg(ms)"
		   << " | " << std::setw(9) << "Median"
		   << " | " << std::setw(9) << "Max(ms)"
		   << " | " << std::setw(9) << "StdDev"
		   << "  |\n";
	}
}

void PrintResultRow(std::ostream &os, const BenchmarkResult &r, size_t index, bool useColor) {
	const char *rowColor = (index % 2 == 0) ? "" : (useColor ? Col(COLOR_GRAY) : "");

	if (useColor) {
		os << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET) << rowColor << std::left << std::setw(24)
		   << r.name.substr(0, 23) << Col(COLOR_RESET) << " │ " << std::right << std::setw(6) << r.measuredCount
		   << " │ " << std::fixed << std::setprecision(3) << std::setw(9) << r.minMs << " │ " << Col(COLOR_GREEN)
		   << std::setw(9) << r.avgMs << Col(COLOR_RESET) << " │ " << std::setw(9) << r.medianMs << " │ "
		   << std::setw(9) << r.maxMs << " │ " << Col(COLOR_YELLOW) << std::setw(9) << r.stddevMs << Col(COLOR_RESET)
		   << Col(COLOR_CYAN) << "  ║\n"
		   << Col(COLOR_RESET);
	} else {
		os << "| " << std::left << std::setw(24) << r.name.substr(0, 23) << " | " << std::right << std::setw(6)
		   << r.measuredCount << " | " << std::fixed << std::setprecision(3) << std::setw(9) << r.minMs << " | "
		   << std::setw(9) << r.avgMs << " | " << std::setw(9) << r.medianMs << " | " << std::setw(9) << r.maxMs
		   << " | " << std::setw(9) << r.stddevMs << "  |\n";
	}
}

} // namespace

// =============================================================================
// BenchmarkRunner
// =============================================================================

void BenchmarkRunner::PrintResults() const {
	std::cout << GetFormattedResults();
}

std::string BenchmarkRunner::GetFormattedResults() const {
	std::ostringstream oss;
	const bool		   useColor = UseColor();

	if (_results.empty()) {
		oss << "[BenchmarkRunner] No benchmark results recorded.\n";
		return oss.str();
	}

	oss << "\n";
	PrintHeader(oss, useColor);
	PrintSeparator(oss, useColor);
	PrintColumnHeaders(oss, useColor);
	PrintSeparator(oss, useColor);

	for (size_t i = 0; i < _results.size(); ++i) {
		PrintResultRow(oss, _results[i], i, useColor);
	}

	PrintFooter(oss, useColor);
	oss << "\n";
	return oss.str();
}

// =============================================================================
// BenchmarkSuite
// =============================================================================

void BenchmarkSuite::Run(const BenchmarkConfig &config) {
	_results.clear();
	_results.reserve(_entries.size());

	BenchmarkRunner runner;
	for (const auto &entry : _entries) {
		runner.RunAndRecord(entry.name, entry.func, config);
	}

	_results = runner.GetResults();
}

void BenchmarkSuite::PrintResults() const {
	std::cout << GetFormattedResults();
}

std::string BenchmarkSuite::GetFormattedResults() const {
	std::ostringstream oss;
	const bool		   useColor = UseColor();

	if (_results.empty()) {
		oss << "[BenchmarkSuite] No results. Call Run() first.\n";
		return oss.str();
	}

	oss << "\n";
	if (useColor) {
		oss << Col(COLOR_MAGENTA) << Col(COLOR_BOLD) << "  Suite: " << _name << Col(COLOR_RESET) << "\n";
	} else {
		oss << "  Suite: " << _name << "\n";
	}

	PrintHeader(oss, useColor);
	PrintSeparator(oss, useColor);
	PrintColumnHeaders(oss, useColor);
	PrintSeparator(oss, useColor);

	for (size_t i = 0; i < _results.size(); ++i) {
		PrintResultRow(oss, _results[i], i, useColor);
	}

	PrintFooter(oss, useColor);

	// Summary line
	double totalMs = 0.0;
	for (const auto &r : _results) {
		totalMs += r.totalMs;
	}
	if (useColor) {
		oss << Col(COLOR_GRAY) << "  Total measurement time: " << Col(COLOR_MAGENTA) << std::fixed
			<< std::setprecision(3) << totalMs << " ms" << Col(COLOR_RESET) << "\n\n";
	} else {
		oss << "  Total measurement time: " << std::fixed << std::setprecision(3) << totalMs << " ms\n\n";
	}

	return oss.str();
}

} // namespace GPU::Benchmark
