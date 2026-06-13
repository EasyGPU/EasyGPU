/**
 * @file main.cpp
 * @brief Minimal self-attention model trained with GPU automatic differentiation.
 *
 * Implements a single-head self-attention mechanism over 2 positions with
 * learnable bias terms and ReLU activation. Each GPU thread processes one
 * training example. Shared weights (all threads read buf[0]) — gradients
 * are summed on CPU.
 *
 * Model:  input(2 pos x 1D) → QKV proj (with bias) → dot-product attention
 *         → ReLU → output linear → MSE loss
 *
 * Demonstrates the AD API with a non-trivial deep learning architecture.
 */

#include <AD/ADKernel.h>
#include <GPU.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::AD;

int main() {
	try {
		std::printf("=== GPU Self-Attention Training with AD ===\n\n");

		constexpr size_t   N		 = 2560;
		constexpr int	   groupSize = 256;
		constexpr int	   groups	 = static_cast<int>(N / groupSize);

		// =========================================================================
		// Generate synthetic data: 2 positions x 1D, target = 1 if x0*x1 > 0
		// =========================================================================
		std::vector<float> x_data(N * 2);
		std::vector<float> y_data(N);
		std::mt19937	   rng(42);
		for (size_t i = 0; i < N; i++) {
			float x0		  = std::normal_distribution<float>(0.0f, 1.0f)(rng);
			float x1		  = std::normal_distribution<float>(0.0f, 1.0f)(rng);
			x_data[i * 2 + 0] = x0;
			x_data[i * 2 + 1] = x1;
			y_data[i]		  = (x0 * x1 > 0.0f) ? 1.0f : 0.0f;
		}

		// =========================================================================
		// Weights: [wq, wk, wv, wo0, wo1, bq, bk, bv, zero]
		// The last element is a constant 0.0 for ReLU (not registered as param).
		// =========================================================================
		constexpr int						  NW = 9;
		std::vector<float>					  W_data(NW);
		std::uniform_real_distribution<float> init_dist(-0.5f, 0.5f);
		for (int i = 0; i < 8; i++)
			W_data[i] = init_dist(rng);
		W_data[8] = 0.0f;

		// =========================================================================
		// GPU buffers (3 forward + 1 gradient = 4 SSBOs, well within limits)
		// =========================================================================
		Buffer<float> buf_x(x_data, BufferMode::Read);
		Buffer<float> buf_y(y_data, BufferMode::Read);
		Buffer<float> buf_W(W_data, BufferMode::ReadWrite);

		// =========================================================================
		// AD Kernel: self-attention with bias + ReLU (8 trainable parameters)
		// =========================================================================
		ADKernel1D	  kernel(
			[&](Var<int> &id) {
				auto	   x_ref  = buf_x.Bind();
				auto	   y_ref  = buf_y.Bind();
				auto	   W_ref  = buf_W.Bind();

				auto	   x0	  = x_ref[id * 2 + 0];
				auto	   x1	  = x_ref[id * 2 + 1];
				auto	   y_true = y_ref[id];

				auto	   wq	  = W_ref[0];
				auto	   wk	  = W_ref[1];
				auto	   wv	  = W_ref[2];
				auto	   wo0	  = W_ref[3];
				auto	   wo1	  = W_ref[4];
				auto	   bq	  = W_ref[5];
				auto	   bk	  = W_ref[6];
				auto	   bv	  = W_ref[7];
				auto	   zero	  = W_ref[8];

				// --- Q, K, V projections with bias ---
				Var<float> q0, q1, k0, k1, v0, v1;
				Var<float> tq0, tq1, tk0, tk1, tv0, tv1;
				tq0 = wq * x0;
				q0	= tq0 + bq;
				tq1 = wq * x1;
				q1	= tq1 + bq;
				tk0 = wk * x0;
				k0	= tk0 + bk;
				tk1 = wk * x1;
				k1	= tk1 + bk;
				tv0 = wv * x0;
				v0	= tv0 + bv;
				tv1 = wv * x1;
				v1	= tv1 + bv;

				// --- Attention scores ---
				Var<float> s00, s01, s10, s11;
				s00 = q0 * k0;
				s01 = q0 * k1;
				s10 = q1 * k0;
				s11 = q1 * k1;

				// --- Softmax row 0 ---
				Var<float> e00, e01, esum0, a00, a01;
				e00	  = Exp(s00);
				e01	  = Exp(s01);
				esum0 = e00 + e01;
				a00	  = e00 / esum0;
				a01	  = e01 / esum0;

				// --- Softmax row 1 ---
				Var<float> e10, e11, esum1, a10, a11;
				e10	  = Exp(s10);
				e11	  = Exp(s11);
				esum1 = e10 + e11;
				a10	  = e10 / esum1;
				a11	  = e11 / esum1;

				// --- Attention output ---
				Var<float> y0_a, y0_b, y0, y1_a, y1_b, y1;
				y0_a = a00 * v0;
				y0_b = a01 * v1;
				y0	 = y0_a + y0_b;
				y1_a = a10 * v0;
				y1_b = a11 * v1;
				y1	 = y1_a + y1_b;

				// --- ReLU activation ---
				Var<float> r0, r1;
				r0 = Max(y0, zero);
				r1 = Max(y1, zero);

				// --- Output projection ---
				Var<float> out0, out1, out;
				out0 = wo0 * r0;
				out1 = wo1 * r1;
				out	 = out0 + out1;

				// --- MSE loss ---
				Var<float> diff, loss;
				diff	 = out - y_true;
				loss	 = diff * diff;

				// --- Register parameters ---
				int pwq	 = AD::Param(wq);
				int pwk	 = AD::Param(wk);
				int pwv	 = AD::Param(wv);
				int pwo0 = AD::Param(wo0);
				int pwo1 = AD::Param(wo1);
				int pbq	 = AD::Param(bq);
				int pbk	 = AD::Param(bk);
				int pbv	 = AD::Param(bv);
				AD::Loss(loss);
				(void)pwq;
				(void)pwk;
				(void)pwv;
				(void)pwo0;
				(void)pwo1;
				(void)pbq;
				(void)pbk;
				(void)pbv;
			},
			N, groupSize);

		// =========================================================================
		// Debug output
		// =========================================================================
		std::string combined = kernel.CombinedCode();
		if (!combined.empty()) {
			std::printf("=== Combined GLSL (first 3000 chars) ===\n%.3000s\n...\n\n", combined.c_str());
		}
		std::printf("Parameters: %zu, Tape entries: %zu\n\n", kernel.ParameterCount(), kernel.Tape().Size());

		// =========================================================================
		// Pipeline compilation check
		// =========================================================================
		std::printf("=== Testing combined pipeline compilation ===\n");
		kernel.Forward(groups, true);
		kernel.Backward(groups, true);
		std::printf("Combined pipeline OK!\n\n");

		// =========================================================================
		// CPU reference for loss computation
		// =========================================================================
		auto computeLoss = [&](const std::vector<float> &W) {
			double total = 0;
			for (size_t i = 0; i < N; i++) {
				float x0 = x_data[i * 2], x1 = x_data[i * 2 + 1];
				float yt = y_data[i];
				float wq = W[0], wk = W[1], wv = W[2], wo0 = W[3], wo1 = W[4];
				float bq = W[5], bk = W[6], bv = W[7];
				float q0 = wq * x0 + bq, q1 = wq * x1 + bq;
				float k0 = wk * x0 + bk, k1 = wk * x1 + bk;
				float v0 = wv * x0 + bv, v1 = wv * x1 + bv;
				float s00 = q0 * k0, s01 = q0 * k1, s10 = q1 * k0, s11 = q1 * k1;
				float e00 = std::exp(s00), e01 = std::exp(s01);
				float es0 = e00 + e01, a00 = e00 / es0, a01 = e01 / es0;
				float e10 = std::exp(s10), e11 = std::exp(s11);
				float es1 = e10 + e11, a10 = e10 / es1, a11 = e11 / es1;
				float y0 = a00 * v0 + a01 * v1, y1 = a10 * v0 + a11 * v1;
				float r0 = y0 > 0 ? y0 : 0, r1 = y1 > 0 ? y1 : 0;
				float out	= wo0 * r0 + wo1 * r1;
				float diff	= out - yt;
				total	   += diff * diff;
			}
			return total;
		};

		// =========================================================================
		// Finite difference gradient verification
		// =========================================================================
		std::printf("=== Gradient Verification ===\n");

		auto sumGrad = [&](int pidx) {
			auto  g	  = kernel.Gradient(pidx);
			float sum = 0;
			for (size_t i = 0; i < N; i++)
				sum += g[i];
			return sum;
		};

		float		eps		= 1e-3f;
		const char *pname[] = {"wq", "wk", "wv", "wo0", "wo1", "bq", "bk", "bv"};
		for (int p = 0; p < 8; p++) {
			auto W_plus = W_data, W_minus = W_data;
			W_plus[p]		  += eps;
			W_minus[p]		  -= eps;

			double loss_plus   = computeLoss(W_plus);
			double loss_minus  = computeLoss(W_minus);

			double fd_grad	   = (loss_plus - loss_minus) / (2.0 * eps);
			float  ad_grad	   = sumGrad(p);

			double err		   = std::abs(fd_grad - ad_grad);
			double denom	   = std::max(std::abs(fd_grad), 1e-6);
			double rel_err	   = err / denom;

			std::printf("  %s: AD=%.6f FD=%.6f rel_err=%.2e %s\n", pname[p], ad_grad, fd_grad, rel_err,
						(rel_err < 0.05 || err < 1e-3) ? "OK" : "WARN");
		}
		std::printf("\n");

		// =========================================================================
		// Training loop with RMSprop-normalized per-parameter updates
		// =========================================================================
		std::printf("=== Training (4000 RMSprop steps) ===\n");
		float lr = 0.05f;

		for (int step = 0; step < 4000; step++) {
			kernel.Forward(groups, true);
			kernel.Backward(groups, true);

			for (int p = 0; p < 8; p++) {
				auto  grad		 = kernel.Gradient(p);
				float total_grad = 0, total_sq = 0;
				for (size_t i = 0; i < N; i++) {
					total_grad += grad[i];
					total_sq   += grad[i] * grad[i];
				}
				float mean_grad	 = total_grad / N;
				float rms		 = std::sqrt(total_sq / N + 1e-8f);
				W_data[p]		-= lr * mean_grad / rms;
			}
			buf_W.Upload(W_data);

			if (step % 400 == 0) {
				double loss_val = computeLoss(W_data);
				std::printf("  Step %3d: loss=%.6f  W=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n", step, loss_val / N,
							W_data[0], W_data[1], W_data[2], W_data[3], W_data[4], W_data[5], W_data[6], W_data[7]);
			}
		}

		// =========================================================================
		// Final evaluation
		// =========================================================================
		double final_loss = computeLoss(W_data) / N;
		std::printf("\n=== Final Results ===\n  Final loss: %.6f\n", final_loss);

		int correct = 0;
		for (size_t i = 0; i < N; i++) {
			float x0 = x_data[i * 2], x1 = x_data[i * 2 + 1];
			float yt = y_data[i];
			float wq = W_data[0], wk = W_data[1], wv = W_data[2], wo0 = W_data[3], wo1 = W_data[4];
			float bq = W_data[5], bk = W_data[6], bv = W_data[7];
			float q0 = wq * x0 + bq, q1 = wq * x1 + bq;
			float k0 = wk * x0 + bk, k1 = wk * x1 + bk;
			float v0 = wv * x0 + bv, v1 = wv * x1 + bv;
			float s00 = q0 * k0, s01 = q0 * k1, s10 = q1 * k0, s11 = q1 * k1;
			float e00 = std::exp(s00), e01 = std::exp(s01);
			float es0 = e00 + e01, a00 = e00 / es0, a01 = e01 / es0;
			float e10 = std::exp(s10), e11 = std::exp(s11);
			float es1 = e10 + e11, a10 = e10 / es1, a11 = e11 / es1;
			float y0 = a00 * v0 + a01 * v1, y1 = a10 * v0 + a11 * v1;
			float r0 = y0 > 0 ? y0 : 0, r1 = y1 > 0 ? y1 : 0;
			float out  = wo0 * r0 + wo1 * r1;
			int	  pred = (out > 0.5f) ? 1 : 0;
			if (pred == (int)yt)
				correct++;
		}
		float acc = 100.0f * correct / N;
		std::printf("  Accuracy: %d/%zu (%.1f%%)\n", correct, N, acc);

		if (acc > 60.0f) {
			std::printf("\n*** TRAINING SUCCESSFUL ***\n");
			return 0;
		} else {
			std::printf("\n*** Accuracy below threshold ***\n");
			return 1;
		}
	} catch (const std::exception &e) {
		std::printf("EXCEPTION: %s\n", e.what());
		return 1;
	} catch (...) {
		std::printf("UNKNOWN EXCEPTION\n");
		return 1;
	}
}
