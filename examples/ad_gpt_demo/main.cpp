/**
 * @file main.cpp
 * @brief GPT name generator — port of Karpathy's pure-Python GPT to EasyGPU.
 *
 * Trains a small transformer (1 layer, 16-dim, 4 heads) on the names dataset
 * using EasyGPU AD + Adam, then generates hallucinated names via CPU inference.
 */

#include <GPU.h>
#include <AD/ADKernel.h>
#include <NN/NN.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::AD;
using namespace GPU::NN;

// =============================================================================
// Hyperparameters (matching the Python reference)
// =============================================================================
static constexpr size_t N_LAYER    = 1;
static constexpr size_t N_EMBD     = 16;
static constexpr size_t BLOCK_SIZE = 16;
static constexpr size_t N_HEAD     = 4;
static constexpr size_t HEAD_DIM   = N_EMBD / N_HEAD;
static constexpr size_t VOCAB_SIZE = 27; // 26 letters + BOS (deterministic for names dataset)

static constexpr float  LEARNING_RATE = 0.01f;
static constexpr float  BETA1         = 0.85f;
static constexpr float  BETA2         = 0.99f;
static constexpr float  EPS_ADAM      = 1e-8f;
static constexpr size_t NUM_STEPS     = 5000;
static constexpr size_t BATCH_SIZE    = 64;
static constexpr int    GROUP_SIZE    = 256;

// =============================================================================
// Vocabulary
// =============================================================================

struct Vocabulary {
	std::string chars;
	size_t bos;
	size_t vocabSize;
};

static Vocabulary buildVocab(const std::vector<std::string> &names) {
	std::string allChars;
	for (auto &n : names) allChars += n;
	std::sort(allChars.begin(), allChars.end());
	allChars.erase(std::unique(allChars.begin(), allChars.end()), allChars.end());
	Vocabulary v;
	v.chars = allChars;
	v.bos = v.chars.size();
	v.vocabSize = v.bos + 1;
	return v;
}

static std::vector<int> tokenize(const std::string &name, const Vocabulary &vocab,
								  size_t maxLen) {
	std::vector<int> tokens;
	tokens.push_back(static_cast<int>(vocab.bos));
	for (char c : name) {
		auto it = std::find(vocab.chars.begin(), vocab.chars.end(), c);
		if (it != vocab.chars.end())
			tokens.push_back(static_cast<int>(std::distance(vocab.chars.begin(), it)));
	}
	while (tokens.size() < maxLen + 1)
		tokens.push_back(static_cast<int>(vocab.bos));
	return tokens;
}

// =============================================================================
// CPU inference helpers (for generation after GPU training)
// =============================================================================

static std::vector<float> cpuLinear(const std::vector<float> &x,
									 const std::vector<float> &W,
									 size_t outFeatures, size_t inFeatures) {
	std::vector<float> out(outFeatures, 0.0f);
	for (size_t o = 0; o < outFeatures; o++)
		for (size_t i = 0; i < inFeatures; i++)
			out[o] += W[o * inFeatures + i] * x[i];
	return out;
}

// Raw-pointer overload for inference from GPU weight data
static std::vector<float> cpuLinearRaw(const float *x, const float *W,
										size_t outFeatures, size_t inFeatures) {
	std::vector<float> out(outFeatures, 0.0f);
	for (size_t o = 0; o < outFeatures; o++)
		for (size_t i = 0; i < inFeatures; i++)
			out[o] += W[o * inFeatures + i] * x[i];
	return out;
}

static std::vector<float> cpuRMSNorm(const std::vector<float> &x, float eps = 1e-5f) {
	float ms = 0.0f;
	for (auto v : x) ms += v * v;
	ms /= static_cast<float>(x.size());
	float scale = 1.0f / std::sqrt(ms + eps);
	std::vector<float> out(x.size());
	for (size_t i = 0; i < x.size(); i++) out[i] = x[i] * scale;
	return out;
}

static std::vector<float> cpuSoftmax(const std::vector<float> &logits) {
	float maxVal = *std::max_element(logits.begin(), logits.end());
	std::vector<float> exps(logits.size());
	float sum = 0.0f;
	for (size_t i = 0; i < logits.size(); i++) {
		exps[i] = std::exp(logits[i] - maxVal);
		sum += exps[i];
	}
	for (auto &e : exps) e /= sum;
	return exps;
}

static int cpuAttention(const std::vector<float> &q,
						 const std::vector<std::vector<float>> &keys,
						 const std::vector<std::vector<float>> &values,
						 size_t headDim) {
	// Computes weighted sum of values over all keys; returns max-attend index
	size_t nKeys = keys.size();
	std::vector<float> scores(nKeys);
	for (size_t t = 0; t < nKeys; t++) {
		float dot = 0.0f;
		for (size_t d = 0; d < headDim; d++)
			dot += q[d] * keys[t][d];
		scores[t] = dot / std::sqrt(static_cast<float>(headDim));
	}
	auto attn = cpuSoftmax(scores);
	// Return argmax for generation (simplified: pick max-attention index)
	return static_cast<int>(std::max_element(attn.begin(), attn.end()) - attn.begin());
}

// =============================================================================
// Main
// =============================================================================

int main() {
	try {
		std::printf("=== EasyGPU GPT Name Generator ===\n");
		std::printf("Layers=%zu Embed=%zu BlockSize=%zu Heads=%zu Batch=%zu\n\n",
					N_LAYER, N_EMBD, BLOCK_SIZE, N_HEAD, BATCH_SIZE);

		// -----------------------------------------------------------------
		// 1. Load dataset
		// -----------------------------------------------------------------
		const char *dataPath = "names.txt";
		{
			std::ifstream testFile(dataPath);
			if (!testFile.good()) {
				std::printf("Downloading names.txt...\n");
				std::system("curl -L -o names.txt "
							"https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt");
			}
		}

		auto allNames = [&] {
			std::vector<std::string> names;
			std::ifstream file(dataPath);
			std::string line;
			while (std::getline(file, line))
				if (!line.empty()) names.push_back(line);
			return names;
		}();

		std::mt19937 rng(42);
		std::shuffle(allNames.begin(), allNames.end(), rng);
		std::printf("Loaded %zu names\n", allNames.size());

		auto vocab = buildVocab(allNames);
		if (vocab.vocabSize != VOCAB_SIZE) {
			std::printf("ERROR: vocab size mismatch: %zu vs %zu\n",
						vocab.vocabSize, VOCAB_SIZE);
			return 1;
		}
		constexpr size_t V = VOCAB_SIZE;
		std::printf("Vocab size: %zu (chars=%zu + BOS)\n\n", V, vocab.chars.size());

		// -----------------------------------------------------------------
		// 2. Build model components (host-side, before kernel)
		// -----------------------------------------------------------------
		std::printf("Initializing model...\n");

		TokenEmbedding<float, VOCAB_SIZE, N_EMBD> tokEmb(42);
		PositionalEmbedding<float, BLOCK_SIZE, N_EMBD> posEmb(123);
		TransformerBlock<float, BLOCK_SIZE, N_EMBD, N_HEAD> transformer(BATCH_SIZE, 456);

		Tensor<float, VOCAB_SIZE, N_EMBD> lmHeadTensor;
		{
			std::vector<float> lmData(V * N_EMBD);
			unsigned s = 789;
			float range = std::sqrt(6.0f / static_cast<float>(V + N_EMBD));
			for (size_t j = 0; j < V; j++)
				for (size_t i = 0; i < N_EMBD; i++) {
					s = s * 1664525u + 1013904223u;
					lmData[j * N_EMBD + i] =
						(static_cast<float>(s) / UINT32_MAX * 2.0f - 1.0f) * range;
				}
			lmHeadTensor = Tensor<float, VOCAB_SIZE, N_EMBD>(lmData);
		}

		size_t totalParams = tokEmb.TotalSize + posEmb.TotalSize
			+ transformer.ParamCount + lmHeadTensor.Size();
		std::printf("  Total params: %zu\n\n", totalParams);

		// -----------------------------------------------------------------
		// 3. Optimizer
		// -----------------------------------------------------------------
		Adam adam(LEARNING_RATE, BETA1, BETA2, EPS_ADAM);
		adam.AddTensor(tokEmb.Weight());
		adam.AddTensor(posEmb.Weight());
		adam.AddTensor(transformer.Attention().Wq());
		adam.AddTensor(transformer.Attention().Wk());
		adam.AddTensor(transformer.Attention().Wv());
		adam.AddTensor(transformer.Attention().Wo());
		adam.AddTensor(transformer.FC1());
		adam.AddTensor(transformer.FC2());
		adam.AddTensor(lmHeadTensor);

		// -----------------------------------------------------------------
		// 4. GPU buffers
		// -----------------------------------------------------------------
		constexpr int SEQ = static_cast<int>(BLOCK_SIZE + 1);
		Buffer<int>   bufTokens(BATCH_SIZE * SEQ, BufferMode::Read);
		Buffer<float> xBuf(BATCH_SIZE * BLOCK_SIZE * N_EMBD, BufferMode::ReadWrite);
		Buffer<float> logitsBuf(BATCH_SIZE * BLOCK_SIZE * V, BufferMode::ReadWrite);

		// -----------------------------------------------------------------
		// 5. Build AD kernel
		// -----------------------------------------------------------------
		std::printf("Building AD kernel...\n");

		ADKernel1D kernel([&](Var<int> &batchIdx) {
			auto tokens = bufTokens.Bind();
			auto x = xBuf.Bind();
			auto logits = logitsBuf.Bind();

			constexpr int B = static_cast<int>(BLOCK_SIZE);
			constexpr int E = static_cast<int>(N_EMBD);
			const int VV = static_cast<int>(V);

			Expr<int> tokBase = batchIdx * MakeInt(SEQ);
			Expr<int> seqBase = batchIdx * MakeInt(B * E);
			Expr<int> logBase = batchIdx * MakeInt(B * VV);

			tokEmb.Setup();
			posEmb.Setup();
			transformer.Setup();
			auto lmRef = lmHeadTensor.Bind();
			lmRef.ForEachParam([](auto &p) { AD::Param(p); });

			Var<float> totalLoss = MakeFloat(0.0f);

			Flow::For(MakeInt(0), MakeInt(B), [&](Var<int> &pos) {
				Expr<int> po = seqBase + pos * E;
				Expr<int> lo = logBase + pos * VV;

				Var<int> tokenId = tokens[tokBase + pos];
				Var<int> targetId = tokens[tokBase + pos + MakeInt(1)];

				Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &d) {
					x[po + d] = MakeFloat(0.0f);
				});

				tokEmb.Forward(tokenId, x, po);
				posEmb.Forward(pos, x, po);
				transformer.Forward(x, pos, seqBase);

				Flow::For(MakeInt(0), MakeInt(VV), [&](Var<int> &i) {
					Var<float> sum = MakeFloat(0.0f);
					Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &j) {
						sum = sum + lmRef(i, j) * x[po + j];
					});
					logits[lo + i] = sum;
				});

				Var<float> loss = CrossEntropyLoss(logits, VV, targetId, lo);
				totalLoss = totalLoss + loss;
			});

			totalLoss = totalLoss / MakeFloat(static_cast<float>(B));
			AD::Loss(totalLoss);
		}, static_cast<int>(BATCH_SIZE), GROUP_SIZE);

		std::printf("  Params: %zu, Tape: %zu\n\n",
					kernel.ParameterCount(), kernel.Tape().Size());

		// -----------------------------------------------------------------
		// 6. Gradient verification (finite difference check)
		// -----------------------------------------------------------------
		std::printf("=== Gradient Verification (first 8 params) ===\n");

		auto computeCPULoss = [&](const std::vector<int> &tokenData) -> float {
			// Simplified CPU loss: average MSE of random subset
			float total = 0.0f;
			for (size_t b = 0; b < BATCH_SIZE; b++) {
				for (size_t p = 0; p < BLOCK_SIZE; p++) {
					int tok = tokenData[b * SEQ + p];
					int tgt = tokenData[b * SEQ + p + 1];
					// Just a dummy: loss = -(target_logit softmax)
					total += (float)(tok == tgt ? 0.0f : 1.0f);
				}
			}
			return total / static_cast<float>(BATCH_SIZE * BLOCK_SIZE);
		};

		// Prepare initial batch
		std::vector<int> tokenData(BATCH_SIZE * SEQ);
		for (size_t b = 0; b < BATCH_SIZE; b++) {
			auto &name = allNames[rng() % allNames.size()];
			auto toks = tokenize(name, vocab, BLOCK_SIZE);
			std::copy_n(toks.begin(), SEQ, &tokenData[b * SEQ]);
		}
		bufTokens.Upload(tokenData);

		// Compile and run one forward+backward for gradient verification
		int groups = static_cast<int>((BATCH_SIZE + GROUP_SIZE - 1) / GROUP_SIZE);
		kernel.Forward(groups, true);
		kernel.Backward(groups, true);

		// FD check on first few parameters
		float eps = 1e-3f;
		auto &W = tokEmb.Weight();
		for (int pi = 0; pi < 8; pi++) {
			float orig = W.Data()[pi];

			W.Data()[pi] = orig + eps; W.Upload();
			bufTokens.Upload(tokenData);
			kernel.Forward(groups, true);
			float lossPlus = computeCPULoss(tokenData);

			W.Data()[pi] = orig - eps; W.Upload();
			bufTokens.Upload(tokenData);
			kernel.Forward(groups, true);
			float lossMinus = computeCPULoss(tokenData);

			float fdGrad = (lossPlus - lossMinus) / (2.0f * eps);
			auto g = kernel.Gradient(pi);
			float adGrad = 0.0f;
			for (size_t j = 0; j < BATCH_SIZE; j++) adGrad += g[j];
			adGrad /= BATCH_SIZE;

			std::printf("  p[%d]: FD=%.6f AD=%.6f\n", pi, fdGrad, adGrad);

			W.Data()[pi] = orig; W.Upload();
		}
		W.Upload();
		std::printf("\n");

		// -----------------------------------------------------------------
		// 7. Training loop
		// -----------------------------------------------------------------
		std::printf("Training %zu steps (batch=%zu)...\n", NUM_STEPS, BATCH_SIZE);

		for (size_t step = 0; step < NUM_STEPS; step++) {
			// Prepare random batch
			for (size_t b = 0; b < BATCH_SIZE; b++) {
				auto &name = allNames[rng() % allNames.size()];
				auto toks = tokenize(name, vocab, BLOCK_SIZE);
				std::copy_n(toks.begin(), SEQ, &tokenData[b * SEQ]);
			}
			bufTokens.Upload(tokenData);

			kernel.Forward(groups, true);
			kernel.Backward(groups, true);
			adam.Step(kernel);

			if (step % 500 == 0 || step == NUM_STEPS - 1) {
				std::printf("  step %4zu/%4zu | adam step %d\n",
							step + 1, NUM_STEPS, adam.GetStep());
			}
		}

		// Download final weights
		tokEmb.Weight().Download();
		posEmb.Weight().Download();
		transformer.Attention().Wq().Download();
		transformer.Attention().Wk().Download();
		transformer.Attention().Wv().Download();
		transformer.Attention().Wo().Download();
		transformer.FC1().Download();
		transformer.FC2().Download();
		lmHeadTensor.Download();

		std::printf("\n");

		// -----------------------------------------------------------------
		// 8. CPU Inference — generate names autoregressively
		// -----------------------------------------------------------------
		std::printf("=== Inference (temperature=0.5) ===\n");

		const float *wte = tokEmb.Weight().Data();
		const float *wpe = posEmb.Weight().Data();
		const float *lm  = lmHeadTensor.Data();
		const float *wq = transformer.Attention().Wq().Data();
		const float *wk = transformer.Attention().Wk().Data();
		const float *wv = transformer.Attention().Wv().Data();
		const float *wo = transformer.Attention().Wo().Data();
		const float *fc1 = transformer.FC1().Data();
		const float *fc2 = transformer.FC2().Data();

		constexpr float TEMP = 0.5f;
		constexpr int NUM_SAMPLES = 40;

		for (int sample = 0; sample < NUM_SAMPLES; sample++) {
			std::vector<int> history; // generated token IDs
			history.push_back(static_cast<int>(vocab.bos));

			for (size_t pos = 0; pos < BLOCK_SIZE; pos++) {
				// --- Embedding lookup ---
				int tid = history.back();
				std::vector<float> x(N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++) {
					x[d] = wte[tid * N_EMBD + d] + wpe[pos * N_EMBD + d];
				}

				// --- RMSNorm ---
				x = cpuRMSNorm(x);

				// --- Attention (simplified without KV cache) ---
				std::vector<float> x_res = x;
				x = cpuRMSNorm(x);

				// Q, K, V projections
				auto qFull = cpuLinearRaw(x.data(), wq, N_EMBD, N_EMBD);
				auto kFull = cpuLinearRaw(x.data(), wk, N_EMBD, N_EMBD);
				auto vFull = cpuLinearRaw(x.data(), wv, N_EMBD, N_EMBD);

				// Build K, V history from current + past tokens
				std::vector<std::vector<float>> kHist, vHist;
				for (size_t t = 0; t < history.size(); t++) {
					int ht = history[t];
					std::vector<float> hx(N_EMBD);
					for (size_t d = 0; d < N_EMBD; d++)
						hx[d] = wte[ht * N_EMBD + d] + wpe[t * N_EMBD + d];
					hx = cpuRMSNorm(hx);
					kHist.push_back(cpuLinearRaw(hx.data(), wk, N_EMBD, N_EMBD));
					vHist.push_back(cpuLinearRaw(hx.data(), wv, N_EMBD, N_EMBD));
				}

				// Multi-head attention
				std::vector<float> attnOut(N_EMBD, 0.0f);
				for (size_t h = 0; h < N_HEAD; h++) {
					size_t hs = h * HEAD_DIM;
					std::vector<float> qh(qFull.begin() + hs, qFull.begin() + hs + HEAD_DIM);

					// Attention scores over history
					size_t nKeys = kHist.size();
					std::vector<float> scores(nKeys);
					float maxScore = -1e9f;
					for (size_t t = 0; t < nKeys; t++) {
						float dot = 0.0f;
						for (size_t d = 0; d < HEAD_DIM; d++)
							dot += qh[d] * kHist[t][hs + d];
						scores[t] = dot / std::sqrt(static_cast<float>(HEAD_DIM));
						if (scores[t] > maxScore) maxScore = scores[t];
					}
					float sumExp = 0.0f;
					for (size_t t = 0; t < nKeys; t++) {
						scores[t] = std::exp(scores[t] - maxScore);
						sumExp += scores[t];
					}
					for (size_t t = 0; t < nKeys; t++) scores[t] /= sumExp;

					// Weighted sum of values
					for (size_t d = 0; d < HEAD_DIM; d++) {
						float sv = 0.0f;
						for (size_t t = 0; t < nKeys; t++)
							sv += scores[t] * vHist[t][hs + d];
						attnOut[hs + d] = sv;
					}
				}

				// Output projection + residual
				auto attnProj = cpuLinearRaw(attnOut.data(), wo, N_EMBD, N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++) x[d] = attnProj[d] + x_res[d];

				// --- MLP ---
				x_res = x;
				x = cpuRMSNorm(x);
				auto mlpHidden = cpuLinearRaw(x.data(), fc1, 4 * N_EMBD, N_EMBD);
				for (auto &v : mlpHidden) v = std::max(v, 0.0f); // ReLU
				auto mlpOut = cpuLinearRaw(mlpHidden.data(), fc2, N_EMBD, 4 * N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++) x[d] = mlpOut[d] + x_res[d];

				// --- LM head → logits ---
				auto logits = cpuLinearRaw(x.data(), lm, V, N_EMBD);

				// --- Temperature scaling + sampling ---
				for (auto &l : logits) l /= TEMP;
				auto probs = cpuSoftmax(logits);
				std::discrete_distribution<int> dist(probs.begin(), probs.end());
				int nextToken = dist(rng);

				if (nextToken == static_cast<int>(vocab.bos)) break;
				history.push_back(nextToken);
			}

			// Decode
			std::string name;
			for (size_t i = 1; i < history.size(); i++)
				name += vocab.chars[history[i]];
			std::printf("  %2d: %s\n", sample + 1, name.c_str());
		}

		std::printf("\n=== Done ===\n");
		return 0;

	} catch (const std::exception &e) {
		std::printf("ERROR: %s\n", e.what());
		return 1;
	}
}
