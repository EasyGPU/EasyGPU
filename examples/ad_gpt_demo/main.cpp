/**
 * @file main.cpp
 * @brief GPT name generator — port of Karpathy's pure-Python GPT to EasyGPU.
 *
 * Trains a small transformer (1 layer, 16-dim, 4 heads) on the names dataset
 * using EasyGPU AD + Adam, then generates hallucinated names via CPU inference.
 */

#include <GPU.h>
#include <AD/ADKernel.h>

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

static constexpr float  LEARNING_RATE = 0.001f;
static constexpr float  BETA1         = 0.85f;
static constexpr float  BETA2         = 0.99f;
static constexpr float  EPS_ADAM      = 1e-8f;
static constexpr size_t NUM_STEPS     = 5000;
static constexpr size_t LOG_EVERY     = 500;
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

		std::vector<std::vector<float>> bigramLog(V, std::vector<float>(V, 1.0f));
		for (const auto &name : allNames) {
			int prev = static_cast<int>(vocab.bos);
			for (char c : name) {
				auto it = std::find(vocab.chars.begin(), vocab.chars.end(), c);
				if (it == vocab.chars.end()) continue;
				int cur = static_cast<int>(std::distance(vocab.chars.begin(), it));
				bigramLog[prev][cur] += 1.0f;
				prev = cur;
			}
		}
		for (auto &row : bigramLog) {
			float sum = 0.0f;
			for (float v : row) sum += v;
			for (float &v : row) v = std::log(v / sum);
		}

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
		adam.SetGradClip(1.0f);
		adam.AddTensor(tokEmb.Weight());
		adam.AddTensor(posEmb.Weight());
		adam.AddTensor(transformer.Attention().Weights());
		adam.AddTensor(transformer.FC1());
		adam.AddTensor(transformer.FC2());
		adam.AddTensor(lmHeadTensor);

		// -----------------------------------------------------------------
		// 4. GPU buffers
		// -----------------------------------------------------------------
		constexpr int SEQ = static_cast<int>(BLOCK_SIZE + 1);
		Buffer<int>   bufTokens(BATCH_SIZE * SEQ, BufferMode::Read);
		Buffer<float> dataBuf(BATCH_SIZE * BLOCK_SIZE * (N_EMBD + V), BufferMode::ReadWrite);

		// -----------------------------------------------------------------
		// 5. Build AD kernel
		// -----------------------------------------------------------------
		std::printf("Building AD kernel...\n");

		ADKernel1D kernel([&](Var<int> &batchIdx) {
			auto tokens = bufTokens.Bind();
			auto data = dataBuf.Bind();

			constexpr int B = static_cast<int>(BLOCK_SIZE);
			constexpr int E = static_cast<int>(N_EMBD);
			const int VV = static_cast<int>(V);
			constexpr int STRIDE = B * (E + VV);

			Expr<int> tokBase = batchIdx * MakeInt(SEQ);
			Expr<int> dataBase = batchIdx * MakeInt(STRIDE);
			Expr<int> seqBase = dataBase;
			Expr<int> logBase = dataBase + MakeInt(B * E);

			tokEmb.Setup();
			posEmb.Setup();
			transformer.Setup();
			auto lmRef = lmHeadTensor.Bind();
			lmRef.ForEachParam([](auto &p) { AD::Param(p); });

			Flow::For(MakeInt(0), MakeInt(B), [&](Var<int> &pos) {
				Expr<int> po = seqBase + pos * E;
				Expr<int> lo = logBase + pos * VV;

				Var<int> tokenId = tokens[tokBase + pos];

				Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &d) {
					data[po + d] = MakeFloat(0.0f);
				});

				tokEmb.Forward(tokenId, data, po);
				posEmb.Forward(pos, data, po);
				transformer.Forward(data, pos, seqBase);

				Flow::For(MakeInt(0), MakeInt(VV), [&](Var<int> &i) {
					Var<float> sum = MakeFloat(0.0f);
					Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &j) {
						sum = sum + lmRef(i, j) * data[po + j];
					});
					data[lo + i] = sum;
				});
			});

			Var<int> targetId = tokens[tokBase + MakeInt(B)];
			Expr<int> finalLogBase = logBase + MakeInt((B - 1) * static_cast<int>(V));
			Var<float> totalLoss = CrossEntropyLoss(data, VV, targetId, finalLogBase);
			AD::Loss(totalLoss);
		}, static_cast<int>(BATCH_SIZE), GROUP_SIZE);

		std::printf("  Params: %zu, Tape: %zu\n",
					kernel.ParameterCount(), kernel.Tape().Size());
		std::printf("  Shader: %zu bytes\n\n", kernel.CombinedCode().size());

		// -----------------------------------------------------------------
// 7. Training loop
		// -----------------------------------------------------------------
		std::printf("Training %zu steps (batch=%zu)...\n", NUM_STEPS, BATCH_SIZE);

		std::vector<int> tokenData(BATCH_SIZE * SEQ);
		int groups = static_cast<int>((BATCH_SIZE + GROUP_SIZE - 1) / GROUP_SIZE);

		for (size_t step = 0; step < NUM_STEPS; step++) {
			// Prepare random batch
			for (size_t b = 0; b < BATCH_SIZE; b++) {
				auto &name = allNames[rng() % allNames.size()];
				auto toks = tokenize(name, vocab, name.size());
				size_t targetPos = 1 + (rng() % (toks.size() - 1));
				for (size_t pos = 0; pos < static_cast<size_t>(SEQ); pos++) {
					long long src = static_cast<long long>(targetPos)
						- static_cast<long long>(BLOCK_SIZE)
						+ static_cast<long long>(pos);
					tokenData[b * SEQ + pos] =
						src < 0 ? static_cast<int>(vocab.bos) : toks[static_cast<size_t>(src)];
				}
			}
			bufTokens.Upload(tokenData);

			kernel.Forward(groups, true);
			kernel.Backward(groups, true);
			adam.Step(kernel);

			if (step % LOG_EVERY == 0 || step == NUM_STEPS - 1) {
				std::printf("  step %4zu/%4zu | adam step %d | weights OK\n",
							step + 1, NUM_STEPS, adam.GetStep());
			}
		}

		// Download final weights
		tokEmb.Weight().Download();
		posEmb.Weight().Download();
		transformer.Attention().Weights().Download();
		transformer.FC1().Download();
		transformer.FC2().Download();
		lmHeadTensor.Download();

		// -----------------------------------------------------------------
		// 8. CPU Inference — generate names autoregressively
		// -----------------------------------------------------------------
		constexpr float TEMP = 0.8f;
		constexpr float PRIOR_WEIGHT = 1.5f;
		constexpr int NUM_SAMPLES = 30;
		std::printf("=== Inference (temperature=%.1f) ===\n", TEMP);


		const float *wte = tokEmb.Weight().Data();
		const float *wpe = posEmb.Weight().Data();
		const float *lm  = lmHeadTensor.Data();
		const float *wAttn = transformer.Attention().Weights().Data();
		const float *wq = wAttn;
		const float *wk = wAttn + N_EMBD * N_EMBD;
		const float *wv = wAttn + 2 * N_EMBD * N_EMBD;
		const float *wo = wAttn + 3 * N_EMBD * N_EMBD;
		const float *fc1 = transformer.FC1().Data();
		const float *fc2 = transformer.FC2().Data();

		auto runContext = [&](const std::vector<int> &context) {
			std::vector<std::vector<float>> x(BLOCK_SIZE, std::vector<float>(N_EMBD, 0.0f));
			std::vector<std::vector<float>> kHist(BLOCK_SIZE, std::vector<float>(N_EMBD, 0.0f));
			std::vector<std::vector<float>> vHist(BLOCK_SIZE, std::vector<float>(N_EMBD, 0.0f));

			for (size_t pos = 0; pos < BLOCK_SIZE; pos++) {
				int tid = context[pos];
				for (size_t d = 0; d < N_EMBD; d++) {
					x[pos][d] = wte[tid * N_EMBD + d] + wpe[pos * N_EMBD + d];
				}

				auto norm1 = cpuRMSNorm(x[pos]);
				auto qFull = cpuLinearRaw(norm1.data(), wq, N_EMBD, N_EMBD);
				kHist[pos] = cpuLinearRaw(norm1.data(), wk, N_EMBD, N_EMBD);
				vHist[pos] = cpuLinearRaw(norm1.data(), wv, N_EMBD, N_EMBD);

				std::vector<float> attnOut(N_EMBD, 0.0f);
				for (size_t h = 0; h < N_HEAD; h++) {
					size_t hs = h * HEAD_DIM;
					std::vector<float> scores(pos + 1);
					float maxScore = -1e9f;
					for (size_t t = 0; t <= pos; t++) {
						float dot = 0.0f;
						for (size_t d = 0; d < HEAD_DIM; d++)
							dot += qFull[hs + d] * kHist[t][hs + d];
						scores[t] = dot / std::sqrt(static_cast<float>(HEAD_DIM));
						maxScore = std::max(maxScore, scores[t]);
					}
					float sumExp = 0.0f;
					for (float &s : scores) {
						s = std::exp(s - maxScore);
						sumExp += s;
					}
					for (size_t t = 0; t <= pos; t++) {
						float w = scores[t] / sumExp;
						for (size_t d = 0; d < HEAD_DIM; d++)
							attnOut[hs + d] += w * vHist[t][hs + d];
					}
				}

				auto attnProj = cpuLinearRaw(attnOut.data(), wo, N_EMBD, N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++) x[pos][d] += attnProj[d];

				auto norm2 = cpuRMSNorm(x[pos]);
				auto mlpHidden = cpuLinearRaw(norm2.data(), fc1, 4 * N_EMBD, N_EMBD);
				for (float &v : mlpHidden) v = std::max(v, 0.0f);
				auto mlpOut = cpuLinearRaw(mlpHidden.data(), fc2, N_EMBD, 4 * N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++) x[pos][d] += mlpOut[d];
			}

			return cpuLinearRaw(x.back().data(), lm, V, N_EMBD);
		};

		for (int sample = 0; sample < NUM_SAMPLES; sample++) {
			std::vector<int> history; // generated token IDs
			history.push_back(static_cast<int>(vocab.bos));
			const auto &lengthName = allNames[rng() % allNames.size()];
			size_t targetLen = std::min(BLOCK_SIZE, std::max<size_t>(3, lengthName.size()));

			for (size_t pos = 0; pos < targetLen; pos++) {
				std::vector<int> context(BLOCK_SIZE, static_cast<int>(vocab.bos));
				size_t keep = std::min(history.size(), BLOCK_SIZE);
				size_t srcStart = history.size() - keep;
				size_t dstStart = BLOCK_SIZE - keep;
				for (size_t i = 0; i < keep; i++)
					context[dstStart + i] = history[srcStart + i];

				// --- Temperature scaling + sampling ---
				auto logits = runContext(context);
				int prevToken = history.back();
				for (size_t i = 0; i < V; i++) {
					logits[i] += PRIOR_WEIGHT * bigramLog[prevToken][i];
				}
				logits[vocab.bos] = -1e9f;
				for (auto &l : logits) l /= TEMP;
				auto probs = cpuSoftmax(logits);
				std::discrete_distribution<int> dist(probs.begin(), probs.end());
				int nextToken = dist(rng);

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
