/**
 * KernelProfiler.cpp:
 *      @Descripiton    :   Kernel profiling tool implementation using OpenGL timer queries
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */
 
// Makes MSVC Happy :)
#define _CRT_SECURE_NO_WARNINGS

#include <Kernel/KernelProfiler.h>
#include <Runtime/Context.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <glad/glad.h>

namespace GPU::Kernel {

    // ===================================================================================
    // Singleton
    // ===================================================================================

    KernelProfiler& KernelProfiler::GetInstance() {
        static KernelProfiler instance;
        return instance;
    }

    KernelProfiler::~KernelProfiler() {
        CleanupQueries();
    }

    // ===================================================================================
    // Query Pool Management
    // ===================================================================================

    void KernelProfiler::InitializeQueries() {
        if (!_queryPool.empty()) return;

        _queryPool.resize(MAX_QUERIES);
        glGenQueries(static_cast<GLsizei>(MAX_QUERIES), _queryPool.data());
        
        for (auto query : _queryPool) {
            _availableQueries.push_back(query);
        }
    }

    void KernelProfiler::CleanupQueries() {
        if (!_queryPool.empty()) {
            glDeleteQueries(static_cast<GLsizei>(_queryPool.size()), _queryPool.data());
            _queryPool.clear();
            _availableQueries.clear();
        }
    }

    unsigned int KernelProfiler::AcquireQuery() {
        if (_queryPool.empty()) {
            InitializeQueries();
        }

        if (_availableQueries.empty()) {
            // All queries in use - try to reclaim completed ones
            for (auto it = _queryPool.begin(); it != _queryPool.end(); ++it) {
                GLint available = 0;
                glGetQueryObjectiv(*it, GL_QUERY_RESULT_AVAILABLE, &available);
                if (available) {
                    _availableQueries.push_back(*it);
                }
            }
        }

        if (_availableQueries.empty()) {
            // Still no available queries - return 0 to skip profiling this dispatch
            return 0;
        }

        unsigned int query = _availableQueries.back();
        _availableQueries.pop_back();
        return query;
    }

    void KernelProfiler::ReleaseQuery(unsigned int query) {
        if (query != 0) {
            _availableQueries.push_back(query);
        }
    }

    // ===================================================================================
    // Enable/Disable
    // ===================================================================================

    void KernelProfiler::SetEnabled(bool enabled) {
        if (_enabled == enabled) return;
        
        _enabled = enabled;
        
        if (_enabled) {
            // Ensure OpenGL context is initialized
            Runtime::AutoInitContext();
            InitializeQueries();
        }
    }

    bool KernelProfiler::IsEnabled() const {
        return _enabled;
    }

    // ===================================================================================
    // Recording
    // ===================================================================================

    void KernelProfiler::Clear() {
        _records.clear();
        _stats.clear();
        // Note: We don't delete queries, just clear the data
    }

    unsigned int KernelProfiler::BeginQuery() {
        if (!_enabled) return 0;
        
        Runtime::ContextGuard guard(Runtime::Context::GetInstance());
        
        unsigned int query = AcquireQuery();
        if (query != 0) {
            glBeginQuery(GL_TIME_ELAPSED, query);
        }
        return query;
    }

    void KernelProfiler::EndQuery(unsigned int queryId, const std::string& kernelName,
                                   int groupX, int groupY, int groupZ) {
        if (!_enabled || queryId == 0) return;

        Runtime::ContextGuard guard(Runtime::Context::GetInstance());
        
        glEndQuery(GL_TIME_ELAPSED);

        // Get the query result
        GLuint64 elapsedNanos = 0;
        glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, &elapsedNanos);
        
        double elapsedMs = static_cast<double>(elapsedNanos) / 1'000'000.0;

        // Record the execution
        KernelProfileRecord record;
        record.kernelName = kernelName;
        record.elapsedTimeMs = elapsedMs;
        record.groupX = groupX;
        record.groupY = groupY;
        record.groupZ = groupZ;
        record.timestamp = std::chrono::system_clock::now();
        _records.push_back(record);

        // Update statistics
        auto& stat = _stats[kernelName];
        stat.kernelName = kernelName;
        stat.counter++;
        stat.totalTimeMs += elapsedMs;
        
        if (stat.counter == 1) {
            stat.minTimeMs = elapsedMs;
            stat.maxTimeMs = elapsedMs;
            stat.avgTimeMs = elapsedMs;
        } else {
            stat.minTimeMs = std::min(stat.minTimeMs, elapsedMs);
            stat.maxTimeMs = std::max(stat.maxTimeMs, elapsedMs);
            stat.avgTimeMs = stat.totalTimeMs / stat.counter;
        }

        ReleaseQuery(queryId);
    }

    // ===================================================================================
    // Query Results
    // ===================================================================================

    KernelProfilerQueryResult KernelProfiler::QueryInfo(const std::string& kernelName) const {
        auto it = _stats.find(kernelName);
        if (it != _stats.end()) {
            return it->second;
        }
        return KernelProfilerQueryResult{}; // Return empty result
    }

    double KernelProfiler::GetTotalTime() const {
        double total = 0.0;
        for (const auto& [name, stat] : _stats) {
            total += stat.totalTimeMs;
        }
        return total;
    }

    const std::vector<KernelProfileRecord>& KernelProfiler::GetRecords() const {
        return _records;
    }

    std::vector<KernelProfilerQueryResult> KernelProfiler::GetAllStats() const {
        std::vector<KernelProfilerQueryResult> results;
        results.reserve(_stats.size());
        for (const auto& [name, stat] : _stats) {
            results.push_back(stat);
        }
        return results;
    }

    // ===================================================================================
    // Printing
    // ===================================================================================

    // ANSI color codes for terminal output
    namespace {
        const char* COLOR_RESET   = "\033[0m";
        const char* COLOR_BOLD    = "\033[1m";
        const char* COLOR_CYAN    = "\033[36m";
        const char* COLOR_GREEN   = "\033[32m";
        const char* COLOR_YELLOW  = "\033[33m";
        const char* COLOR_RED     = "\033[31m";
        const char* COLOR_MAGENTA = "\033[35m";
        const char* COLOR_GRAY    = "\033[90m";

        bool UseColor() {
    #ifdef _WIN32
            return false;  // Windows cmd doesn't support ANSI by default
    #else
            return true;
    #endif
        }

        const char* Col(const char* color) {
            return UseColor() ? color : "";
        }
    }

    void KernelProfiler::PrintInfo(const std::string& mode) const {
        if (!_enabled) {
            std::cout << Col(COLOR_YELLOW) << "[KernelProfiler] " << Col(COLOR_RESET) 
                      << "Profiling is disabled. Call " << Col(COLOR_CYAN) 
                      << "EnableKernelProfiler(true)" << Col(COLOR_RESET) << " to enable.\n";
            return;
        }

        if (_records.empty()) {
            std::cout << Col(COLOR_YELLOW) << "[KernelProfiler] " << Col(COLOR_RESET) 
                      << "No kernel executions recorded.\n";
            return;
        }

        std::cout << "\n";
        // Top border
        std::cout << Col(COLOR_CYAN) << "╔══════════════════════════════════════════════════════════════════════════════╗\n" << Col(COLOR_RESET);
        std::cout << Col(COLOR_CYAN) << "║" << Col(COLOR_BOLD) << "                    🚀  Kernel Profiling Results                               " << Col(COLOR_RESET) << Col(COLOR_CYAN) << "║\n" << Col(COLOR_RESET);
        std::cout << Col(COLOR_CYAN) << "╠══════════════════════════════════════════════════════════════════════════════╣\n" << Col(COLOR_RESET);

        if (mode == "trace") {
            // Trace mode - print individual execution records
            std::cout << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET)
                      << Col(COLOR_BOLD) << std::left << std::setw(28) << "Kernel" << Col(COLOR_RESET)
                      << " │ " << std::right << std::setw(10) << "Time(ms)"
                      << " │ " << std::setw(10) << "Groups"
                      << " │ " << std::setw(16) << "Timestamp"
                      << Col(COLOR_CYAN) << "   ║\n" << Col(COLOR_RESET);
            
            std::cout << Col(COLOR_CYAN) << "╠══════════════════════════════════════════════════════════════════════════════╣\n" << Col(COLOR_RESET);

            for (size_t i = 0; i < _records.size(); ++i) {
                const auto& record = _records[i];
                auto time_t = std::chrono::system_clock::to_time_t(record.timestamp);
                auto tm = *std::localtime(&time_t);
                char timeStr[20];
                std::strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &tm);

                std::string groups = std::to_string(record.groupX);
                if (record.groupY > 1 || record.groupZ > 1) {
                    groups += "x" + std::to_string(record.groupY);
                }
                if (record.groupZ > 1) {
                    groups += "x" + std::to_string(record.groupZ);
                }

                // Alternate row colors
                const char* rowColor = (i % 2 == 0) ? "" : Col(COLOR_GRAY);

                std::cout << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET)
                          << rowColor << std::left << std::setw(28) << record.kernelName.substr(0, 27) << Col(COLOR_RESET)
                          << " │ " << std::right << Col(COLOR_GREEN)
                          << std::fixed << std::setprecision(3) << std::setw(10) << record.elapsedTimeMs << Col(COLOR_RESET)
                          << " │ " << std::setw(10) << groups
                          << " │ " << std::setw(16) << timeStr
                          << Col(COLOR_CYAN) << "   ║\n" << Col(COLOR_RESET);
            }
        } else {
            // Default: count mode - print statistics
            std::cout << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET)
                      << Col(COLOR_BOLD) << std::left << std::setw(24) << "Kernel" << Col(COLOR_RESET)
                      << " │ " << std::right << std::setw(6) << "Count"
                      << " │ " << std::setw(9) << "Min(ms)"
                      << " │ " << std::setw(9) << "Avg(ms)"
                      << " │ " << std::setw(9) << "Max(ms)"
                      << " │ " << std::setw(10) << "Total(ms)"
                      << " │ " << std::setw(6) << "%"
                      << Col(COLOR_CYAN) << "  ║\n" << Col(COLOR_RESET);
            
            std::cout << Col(COLOR_CYAN) << "╠══════════════════════════════════════════════════════════════════════════════╣\n" << Col(COLOR_RESET);

            // Sort by total time (descending)
            auto stats = GetAllStats();
            std::sort(stats.begin(), stats.end(), 
                      [](const auto& a, const auto& b) {
                          return a.totalTimeMs > b.totalTimeMs;
                      });

            double totalTime = GetTotalTime();

            for (size_t i = 0; i < stats.size(); ++i) {
                const auto& stat = stats[i];
                double percent = (totalTime > 0) ? (stat.totalTimeMs / totalTime * 100.0) : 0.0;

                // Alternate row colors
                const char* rowColor = (i % 2 == 0) ? "" : Col(COLOR_GRAY);
                
                // Color code percentage
                const char* percentColor = Col(COLOR_RESET);
                if (percent > 50) percentColor = Col(COLOR_RED);
                else if (percent > 20) percentColor = Col(COLOR_YELLOW);
                else if (percent > 5) percentColor = Col(COLOR_GREEN);

                std::cout << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET)
                          << rowColor << std::left << std::setw(24) << stat.kernelName.substr(0, 23) << Col(COLOR_RESET)
                          << " │ " << std::right << std::setw(6) << stat.counter
                          << " │ " << std::fixed << std::setprecision(3) << std::setw(9) << stat.minTimeMs
                          << " │ " << Col(COLOR_GREEN) << std::setw(9) << stat.avgTimeMs << Col(COLOR_RESET)
                          << " │ " << std::setw(9) << stat.maxTimeMs
                          << " │ " << Col(COLOR_MAGENTA) << std::setw(10) << stat.totalTimeMs << Col(COLOR_RESET)
                          << " │ " << percentColor << std::setw(5) << std::setprecision(1) << percent << "%" << Col(COLOR_RESET)
                          << Col(COLOR_CYAN) << "  ║\n" << Col(COLOR_RESET);
            }

            std::cout << Col(COLOR_CYAN) << "╠══════════════════════════════════════════════════════════════════════════════╣\n" << Col(COLOR_RESET);
            std::cout << Col(COLOR_CYAN) << "║ " << Col(COLOR_RESET)
                      << Col(COLOR_BOLD) << std::left << std::setw(24) << "TOTAL" << Col(COLOR_RESET)
                      << " │ " << std::right << std::setw(6) << _records.size()
                      << " │ " << std::setw(9) << ""
                      << " │ " << std::setw(9) << ""
                      << " │ " << std::setw(9) << ""
                      << " │ " << Col(COLOR_MAGENTA) << Col(COLOR_BOLD) << std::setw(10) << std::fixed << std::setprecision(3) << totalTime << Col(COLOR_RESET)
                      << " │ " << std::setw(6) << "100%"
                      << Col(COLOR_CYAN) << "  ║\n" << Col(COLOR_RESET);
        }

        // Bottom border
        std::cout << Col(COLOR_CYAN) << "╚══════════════════════════════════════════════════════════════════════════════╝\n" << Col(COLOR_RESET);
        
        // Tip
        std::cout << Col(COLOR_GRAY) << "  Tip: Use " << Col(COLOR_CYAN) << "PrintKernelProfilerInfo(\"trace\")" 
                  << Col(COLOR_GRAY) << " for execution trace\n" << Col(COLOR_RESET) << "\n";
    }

    std::string KernelProfiler::GetFormattedOutput(const std::string& mode) const {
        std::ostringstream oss;
        
        if (!_enabled) {
            oss << "[KernelProfiler] Profiling is disabled. Call EnableKernelProfiler(true) to enable.\n";
            return oss.str();
        }

        if (_records.empty()) {
            oss << "[KernelProfiler] No kernel executions recorded.\n";
            return oss.str();
        }

        oss << "\n";
        oss << "+------------------------------------------------------------------------------+\n";
        oss << "|                      Kernel Profiling Results                                |\n";
        oss << "+------------------------------------------------------------------------------+\n";

        if (mode == "trace") {
            oss << "| " << std::left << std::setw(28) << "Kernel"
                << " | " << std::right << std::setw(10) << "Time(ms)"
                << " | " << std::setw(10) << "Groups"
                << " | " << std::setw(16) << "Timestamp" << "   |\n";
            oss << "+------------------------------------------------------------------------------+\n";

            for (const auto& record : _records) {
                auto time_t = std::chrono::system_clock::to_time_t(record.timestamp);
                auto tm = *std::localtime(&time_t);
                char timeStr[20];
                std::strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &tm);

                std::string groups = std::to_string(record.groupX);
                if (record.groupY > 1 || record.groupZ > 1) {
                    groups += "x" + std::to_string(record.groupY);
                }
                if (record.groupZ > 1) {
                    groups += "x" + std::to_string(record.groupZ);
                }

                oss << "| " << std::left << std::setw(28) << record.kernelName.substr(0, 27)
                    << " | " << std::right << std::fixed << std::setprecision(3) << std::setw(10) << record.elapsedTimeMs
                    << " | " << std::setw(10) << groups
                    << " | " << std::setw(16) << timeStr << "   |\n";
            }
        } else {
            oss << "| " << std::left << std::setw(24) << "Kernel"
                << " | " << std::right << std::setw(6) << "Count"
                << " | " << std::setw(9) << "Min(ms)"
                << " | " << std::setw(9) << "Avg(ms)"
                << " | " << std::setw(9) << "Max(ms)"
                << " | " << std::setw(10) << "Total(ms)"
                << " | " << std::setw(6) << "%" << "  |\n";
            
            oss << "+------------------------------------------------------------------------------+\n";

            auto stats = GetAllStats();
            std::sort(stats.begin(), stats.end(), 
                      [](const auto& a, const auto& b) {
                          return a.totalTimeMs > b.totalTimeMs;
                      });

            double totalTime = GetTotalTime();

            for (const auto& stat : stats) {
                double percent = (totalTime > 0) ? (stat.totalTimeMs / totalTime * 100.0) : 0.0;

                oss << "| " << std::left << std::setw(24) << stat.kernelName.substr(0, 23)
                    << " | " << std::right << std::setw(6) << stat.counter
                    << " | " << std::fixed << std::setprecision(3) << std::setw(9) << stat.minTimeMs
                    << " | " << std::setw(9) << stat.avgTimeMs
                    << " | " << std::setw(9) << stat.maxTimeMs
                    << " | " << std::setw(10) << stat.totalTimeMs
                    << " | " << std::setw(5) << std::setprecision(1) << percent << "%" << "  |\n";
            }

            oss << "+------------------------------------------------------------------------------+\n";
            oss << "| " << std::left << std::setw(24) << "TOTAL"
                << " | " << std::right << std::setw(6) << _records.size()
                << " | " << std::setw(9) << ""
                << " | " << std::setw(9) << ""
                << " | " << std::setw(9) << ""
                << " | " << std::setw(10) << std::fixed << std::setprecision(3) << totalTime
                << " | " << std::setw(6) << "100%" << "  |\n";
        }

        oss << "+------------------------------------------------------------------------------+\n";
        oss << "  Tip: Use GetFormattedOutput(\"trace\") for execution trace\n\n";
        
        return oss.str();
    }

    // ===================================================================================
    // RAII Scope Helper
    // ===================================================================================

    KernelProfileScope::KernelProfileScope(const std::string& kernelName, int groupX, int groupY, int groupZ)
        : _kernelName(kernelName)
        , _groupX(groupX)
        , _groupY(groupY)
        , _groupZ(groupZ)
        , _queryId(0) {
        _queryId = KernelProfiler::GetInstance().BeginQuery();
    }

    KernelProfileScope::~KernelProfileScope() {
        KernelProfiler::GetInstance().EndQuery(_queryId, _kernelName, _groupX, _groupY, _groupZ);
    }

}
