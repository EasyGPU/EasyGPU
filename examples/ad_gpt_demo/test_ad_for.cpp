/**
 * @file test_ad_for.cpp
 * @brief Minimal test: does AD work with Flow::For inside a kernel?
 *
 * Tests whether gradient tape correctly records operations inside Flow::For.
 */
#include <GPU.h>
#include <AD/ADKernel.h>
#include <NN/NN.h>

#include <cstdio>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Runtime;
using namespace GPU::AD;
using namespace GPU::NN;

int main() {
    try {
        std::printf("=== AD For-Loop Test ===\n");

        // Simple test: y = W*x + b, but with the inner product done via For loop
        // Just like the GPT model uses For loops for attention and MLP

        constexpr int N = 64;
        constexpr int D = 4;
        constexpr int GS = 256;

        std::vector<float> wData(D * D);  // 4x4 weight matrix
        std::vector<float> xData(N * D);   // N samples of 4-dim
        std::vector<float> tData(N);       // target class per sample

        unsigned s = 42;
        for (size_t i = 0; i < D*D; i++) {
            s = s * 1664525u + 1013904223u;
            wData[i] = (float)s / UINT32_MAX * 0.1f;
        }
        for (size_t i = 0; i < N*D; i++) {
            s = s * 1664525u + 1013904223u;
            xData[i] = (float)s / UINT32_MAX;
        }
        for (size_t i = 0; i < N; i++)
            tData[i] = (float)(i % D);

        Buffer<float> bufW(wData, BufferMode::ReadWrite);
        Buffer<float> bufX(xData, BufferMode::Read);
        Buffer<int>   bufT(tData.size());
        {
            std::vector<int> ti(tData.size());
            for (size_t i = 0; i < tData.size(); i++) ti[i] = (int)tData[i];
            bufT.Upload(ti);
        }

        // Compute logits via For loop (like GPT embedding+linear)
        // logits[tid][i] = sum_j W[i][j] * x[tid][j]
        ADKernel1D kernel([&](Var<int> &tid) {
            auto W = bufW.Bind();
            auto X = bufX.Bind();
            auto T = bufT.Bind();

            // Use For loop to compute logits and loss
            Var<float> loss = MakeFloat(0.0f);

            Flow::For(MakeInt(0), MakeInt(D), [&](Var<int> &i) {
                Var<float> logit = MakeFloat(0.0f);
                Flow::For(MakeInt(0), MakeInt(D), [&](Var<int> &j) {
                    logit = logit + W(i, j) * X[tid * D + j];
                });

                // Cross-entropy contribution for this class
                // Use if statement to check if this is the target class
                auto targetIdx = Var<int>(T[tid]);
                // Compare: if i == targetIdx, this is the target logit
                // Simplified: just use MSE for now
                Var<float> target = MakeFloat(0.0f);
                Flow::If(i == targetIdx, [&] { target = MakeFloat(1.0f); });
                Var<float> diff = logit - target;
                loss = loss + diff * diff;
            });

            // Register all weight params as AD params
            for (int ii = 0; ii < (int)(D*D); ii++) {
                auto wi = W(ii / D, ii % D);
                AD::Param(wi);
            }

            AD::Loss(loss);
        }, N, GS);

        std::printf("Params: %zu, Tape: %zu\n",
                    kernel.ParameterCount(), kernel.Tape().Size());

        // Print combined GLSL to check backward pass
        {
            const auto &code = kernel.CombinedCode();
            auto bp = code.find("Backward pass");
            if (bp != std::string::npos) {
                std::printf("--- Backward pass section ---\n");
                size_t end = std::min(bp + 2000, code.size());
                std::printf("%s\n", code.substr(bp, end - bp).c_str());
            }
        }

        // Run and check gradients
        int groups = (N + GS - 1) / GS;
        kernel.Forward(groups, true);
        kernel.Backward(groups, true);

        bool anyNonZero = false;
        for (int pi = 0; pi < (int)kernel.ParameterCount() && pi < 8; pi++) {
            auto g = kernel.Gradient(pi);
            float sum = 0;
            for (auto v : g) sum += v;
            std::printf("  p[%d]: sum=%.6f size=%zu\n", pi, sum, g.size());
            if (std::abs(sum) > 1e-6f) anyNonZero = true;
        }

        if (anyNonZero) {
            std::printf("PASS: At least one gradient is non-zero\n");
        } else {
            std::printf("FAIL: All gradients are zero\n");
        }

        return 0;
    } catch (const std::exception &e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}
