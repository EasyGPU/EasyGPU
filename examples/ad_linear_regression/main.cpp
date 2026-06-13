/**
 * @file main.cpp
 * @brief GPU linear regression with automatic differentiation.
 *
 * Trains N independent linear models y = W*x + b in parallel on GPU.
 * Each thread optimizes its own (W_i, b_i) pair using SGD.
 * Demonstrates the end-to-end AD API: Forward, Backward, Gradient.
 */

#include <AD/ADKernel.h>
#include <GPU.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Runtime;
using namespace GPU::AD;

int main() {
	try {
		std::printf("Starting AD Linear Regression example...\n");
		constexpr size_t   N		 = 2560;
		constexpr int	   groupSize = 256;
		constexpr int	   groups	 = static_cast<int>(N / groupSize);

		// =========================================================================
		// Generate synthetic data: y_true = 2*x + 1 + noise
		// =========================================================================
		std::vector<float> x_data(N), y_data(N);
		std::mt19937	   rng(42);
		for (size_t i = 0; i < N; i++) {
			float xi  = static_cast<float>(i) / static_cast<float>(N);
			x_data[i] = xi;
			y_data[i] = 2.0f * xi + 1.0f + std::normal_distribution<float>(0.0f, 0.05f)(rng);
		}

		// =========================================================================
		// Initialize shared parameters: W = 0.5, b = 0.0
		// =========================================================================
		std::vector<float> W_data = {0.5f};
		std::vector<float> b_data = {0.0f};

		// =========================================================================
		// GPU buffers
		// =========================================================================
		Buffer<float>	   buf_x(x_data, BufferMode::Read);
		Buffer<float>	   buf_y(y_data, BufferMode::Read);
		Buffer<float>	   buf_W(W_data, BufferMode::ReadWrite);
		Buffer<float>	   buf_b(b_data, BufferMode::ReadWrite);

		// =========================================================================
		// AD Kernel: y_pred = W*x + b, loss = (y_pred - y_true)^2
		// =========================================================================
		ADKernel1D		   kernel(
			[&](Var<int> &id) {
				auto	   x_ref  = buf_x.Bind();
				auto	   y_ref  = buf_y.Bind();
				auto	   W_ref  = buf_W.Bind();
				auto	   b_ref  = buf_b.Bind();

				auto	   x	  = x_ref[id];
				auto	   y_true = y_ref[id];
				// Shared weights: all threads read element 0 (constant index)
				auto	   W	  = W_ref[0];
				auto	   b	  = b_ref[0];

				// Use single-operation Var assignments so each Store is a simple
				// binary/unary op with plain variable inputs — tape can record each.
				Var<float> t1;
				t1 = W * x;
				Var<float> y_pred;
				y_pred = t1 + b;
				Var<float> diff;
				diff = y_pred - y_true;
				Var<float> loss;
				loss   = diff * diff;

				int iW = AD::Param(W);
				int ib = AD::Param(b);
				AD::Loss(loss);
				(void)iW;
				(void)ib;
			},
			N, groupSize);

		// =========================================================================
		// Print generated GLSL for inspection
		// =========================================================================
		std::printf("=== Forward GLSL ===\n%s\n\n", kernel.ForwardCode().c_str());
		std::printf("=== Combined (Forward+Backward) GLSL ===\n%s\n\n", kernel.CombinedCode().c_str());
		std::printf("Parameters: %zu, Tape entries: %zu\n\n", kernel.ParameterCount(), kernel.Tape().Size());

		// =========================================================================
		// Run forward + backward on GPU
		// =========================================================================
		kernel.Forward(groups, true);
		kernel.Backward(groups, true);

		// =========================================================================
		// Download gradients and verify against analytical gradients
		// =========================================================================
		auto grad_W = kernel.Gradient(0);
		auto grad_b = kernel.Gradient(1);

		if (grad_W.empty() || grad_b.empty()) {
			std::printf("FAILED: Gradient download returned empty vector\n");
			return 1;
		}

		// Shared weights: each thread computes per-example gradient,
		// stored in grad_W[i] and grad_b[i]. Sum for total gradient.
		float total_dW = 0, total_db = 0;
		float expected_dW = 0, expected_db = 0;
		float W_shared = W_data[0];
		float b_shared = b_data[0];

		for (size_t i = 0; i < N; i++) {
			float xi		= x_data[i];
			float y_pred_i	= W_shared * xi + b_shared;
			float diff_i	= y_pred_i - y_data[i];

			total_dW	   += grad_W[i];
			total_db	   += grad_b[i];
			expected_dW	   += 2.0f * diff_i * xi;
			expected_db	   += 2.0f * diff_i;
		}

		double err_W = std::abs(total_dW - expected_dW);
		double err_b = std::abs(total_db - expected_db);

		std::printf("Gradient verification (N=%zu, shared weights):\n", N);
		std::printf("  total dW: %f (expected %f), err: %e\n", total_dW, expected_dW, err_W);
		std::printf("  total db: %f (expected %f), err: %e\n", total_db, expected_db, err_b);

		bool gradient_ok = (err_W < 1e-3 && err_b < 1e-3);
		if (!gradient_ok) {
			std::printf("  GRADIENT CHECK FAILED\n");
			return 1;
		}
		std::printf("  GRADIENT CHECK PASSED\n");

		// =========================================================================
		// Run SGD steps with shared weights (sum gradients across threads)
		// =========================================================================
		float lr = 0.1f;
		for (int step = 0; step < 200; step++) {
			kernel.Forward(groups, true);
			kernel.Backward(groups, true);

			auto  dW_per_thread = kernel.Gradient(0);
			auto  db_per_thread = kernel.Gradient(1);

			// Sum per-example gradients
			float total_dW = 0, total_db = 0;
			for (size_t i = 0; i < N; i++) {
				total_dW += dW_per_thread[i];
				total_db += db_per_thread[i];
			}

			// SGD update
			W_data[0] -= lr * total_dW / N;
			b_data[0] -= lr * total_db / N;
			buf_W.Upload(W_data);
			buf_b.Upload(b_data);

			if (step % 20 == 0) {
				// Compute current loss on CPU
				double loss = 0.0;
				for (size_t i = 0; i < N; i++) {
					float xi		= x_data[i];
					float y_pred_i	= W_data[0] * xi + b_data[0];
					float diff_i	= y_pred_i - y_data[i];
					loss		   += diff_i * diff_i;
				}
				loss /= N;
				std::printf("  Step %3d: loss=%.6f  W=%.4f  b=%.4f\n", step, loss, W_data[0], b_data[0]);
			}
		}

		std::printf("\nFinal: W=%.4f (target 2.0), b=%.4f (target 1.0)\n", W_data[0], b_data[0]);

		bool converged = (std::abs(W_data[0] - 2.0) < 0.1 && std::abs(b_data[0] - 1.0) < 0.1);
		if (converged) {
			std::printf("TRAINING CONVERGED\n");
		} else {
			std::printf("Training may need more iterations\n");
		}

		return 0;
	} catch (const std::exception &e) {
		std::printf("EXCEPTION: %s\n", e.what());
		return 1;
	} catch (...) {
		std::printf("UNKNOWN EXCEPTION\n");
		return 1;
	}
}
