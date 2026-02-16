/**
 * @file hello_world.cpp
 * @brief First EasyGPU example: Parallel array increment
 * 
 * This example demonstrates the fundamental EasyGPU workflow:
 * 1. Prepare data on the host (CPU)
 * 2. Upload data to the device (GPU)
 * 3. Define and dispatch a kernel
 * 4. Download results back to the host
 * 5. Verify correctness
 * 
 * The kernel adds 1 to each element of an array in parallel.
 */

#include <GPU.h>

#include <cstdio>
#include <numeric>
#include <vector>

int main() {
    // =========================================================================
    // Configuration
    // =========================================================================
    constexpr size_t kElementCount = 25600;
    constexpr size_t kThreadGroupSize = 256;
    constexpr size_t kDispatchGroupCount = 100;  // 100 * 256 = 25600 threads

    // =========================================================================
    // Host Data Preparation
    // =========================================================================
    // Initialize input array with values [1, 2, 3, ..., kElementCount]
    std::vector<int> host_input(kElementCount);
    std::vector<int> host_output(kElementCount);
    std::iota(host_input.begin(), host_input.end(), 1);

    // =========================================================================
    // Device Data Upload
    // =========================================================================
    // Create GPU buffers and upload initial data
    // Buffer<T> automatically handles memory allocation and data transfer
    Buffer<int> device_input(host_input);     // Upload from vector
    Buffer<int> device_output(kElementCount); // Allocate empty buffer

    // =========================================================================
    // Kernel Definition
    // =========================================================================
    // Define a 1D kernel that processes one element per thread
    // The kernel captures device buffers by reference
    Kernel1D kernel("IncrementKernel", [&](Int tid) {
        // Bind buffers to access them in the kernel scope
        auto input = device_input.Bind();
        auto output = device_output.Bind();

        // Guard against out-of-bounds access (though dispatch size matches here)
        If(tid < static_cast<Int>(kElementCount), [&]() {
            output[tid] = input[tid] + 1;
        });
    }, kThreadGroupSize);

    // =========================================================================
    // Kernel Dispatch
    // =========================================================================
    // Dispatch with barrier: host waits for GPU to complete before continuing
    kernel.Dispatch(kDispatchGroupCount, true);

    // =========================================================================
    // Result Verification
    // =========================================================================
    device_output.Download(host_output);

    bool all_correct = true;
    for (size_t i = 0; i < kElementCount; ++i) {
        const int expected = host_input[i] + 1;
        if (host_output[i] != expected) {
            all_correct = false;
            std::printf("Mismatch at index %zu: got %d, expected %d\n",
                        i, host_output[i], expected);
            break;
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    if (all_correct) {
        std::printf("Success! All %zu elements processed correctly.\n",
                    kElementCount);
    } else {
        std::printf("Failed! Result verification encountered errors.\n");
        return 1;
    }

    return 0;
}
