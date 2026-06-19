/**
 * @file main.cpp
 * @brief Character-level GPT poet demo trained on an embedded poetry corpus.
 *
 * This is a continuous-text language-model demo, not a line/name demo:
 * random windows of BLOCK_SIZE characters predict every next character inside
 * the window.
 */

#include <AD/ADKernel.h>
#include <GPU.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Runtime;
using namespace GPU::AD;
using namespace GPU::NN;

static constexpr size_t		 N_EMBD		   = 16;
static constexpr size_t		 BLOCK_SIZE	   = 32;
static constexpr size_t		 N_HEAD		   = 4;
static constexpr size_t		 HEAD_DIM	   = N_EMBD / N_HEAD;
static constexpr size_t		 VOCAB_SIZE	   = 36;

static constexpr float		 LEARNING_RATE = 0.00008f;
static constexpr float		 BETA1		   = 0.85f;
static constexpr float		 BETA2		   = 0.99f;
static constexpr float		 EPS_ADAM	   = 1e-5f;
static constexpr size_t		 NUM_STEPS	   = 1500;
static constexpr size_t		 LOG_EVERY	   = 250;
static constexpr size_t		 BATCH_SIZE	   = 64;
static constexpr int		 GROUP_SIZE	   = 256;

static constexpr const char *VOCAB		   = "\n abcdefghijklmnopqrstuvwxyz',.;:?!-";
constexpr size_t			 vocabLength(const char *s) {
	size_t n = 0;
	while (s[n] != '\0')
		n++;
	return n;
}
static_assert(vocabLength(VOCAB) == VOCAB_SIZE, "VOCAB_SIZE must match VOCAB");

static const char *POETRY_CORPUS = R"POET(
shall i compare thee to a summer's day?
thou art more lovely and more temperate:
rough winds do shake the darling buds of may,
and summer's lease hath all too short a date:
sometime too hot the eye of heaven shines,
and often is his gold complexion dimm'd;
and every fair from fair sometime declines,
by chance or nature's changing course untrimm'd;
but thy eternal summer shall not fade,
nor lose possession of that fair thou ow'st;
nor shall death brag thou wander'st in his shade,
when in eternal lines to time thou grow'st:
so long as men can breathe or eyes can see,
so long lives this, and this gives life to thee.

how do i love thee? let me count the ways.
i love thee to the depth and breadth and height
my soul can reach, when feeling out of sight
for the ends of being and ideal grace.
i love thee to the level of every day's
most quiet need, by sun and candle-light.
i love thee freely, as men strive for right;
i love thee purely, as they turn from praise.

let me not to the marriage of true minds
admit impediments.
love is not love
which alters when it alteration finds,
or bends with the remover to remove:
o no!
it is an ever-fixed mark,
that looks on tempests and is never shaken;
it is the star to every wandering bark,
whose worth's unknown, although his height be taken.
love's not time's fool, though rosy lips and cheeks
within his bending sickle's compass come:
love alters not with his brief hours and weeks,
but bears it out even to the edge of doom.
if this be error and upon me proved,
i never writ, nor no man ever loved.

how do i love thee?
let me count the ways.
i love thee to the depth and breadth and height
my soul can reach, when feeling out of sight
for the ends of being and ideal grace.
i love thee to the level of every day's
most quiet need, by sun and candle-light.
i love thee freely, as men strive for right.
i love thee purely, as they turn from praise.
i love thee with the passion put to use
in my old griefs, and with my childhood's faith.
i love thee with a love i seemed to lose
with my lost saints,
i love thee with the breath,
smiles, tears, of all my life!
and, if god choose,
i shall but love thee better after death.

she walks in beauty, like the night
of cloudless climes and starry skies;
and all that's best of dark and bright
meet in her aspect and her eyes:
thus mellowed to that tender light
which heaven to gaudy day denies.
one shade the more, one ray the less,
had half impaired the nameless grace
which waves in every raven tress,
or softly lightens o'er her face;
where thoughts serenely sweet express,
how pure, how dear their dwelling-place.
and on that cheek, and o'er that brow,
so soft, so calm, yet eloquent,
the smiles that win, the tints that glow,
but tell of days in goodness spent,
a mind at peace with all below,
a heart whose love is innocent!

when you are old and grey and full of sleep,
and nodding by the fire, take down this book,
and slowly read, and dream of the soft look
your eyes had once, and of their shadows deep;
how many loved your moments of glad grace,
and loved your beauty with love false or true,
but one man loved the pilgrim soul in you,
and loved the sorrows of your changing face;
and bending down beside the glowing bars,
murmur, a little sadly, how love fled
and paced upon the mountains overhead
and hid his face amid a crowd of stars.

bright star, would i were steadfast as thou art-
not in lone splendour hung aloft the night
and watching, with eternal lids apart,
like nature's patient, sleepless eremite,
the moving waters at their priestlike task
of pure ablution round earth's human shores,
or gazing on the new soft-fallen mask
of snow upon the mountains and the moors-
no-yet still stedfast, still unchangeable,
pillow'd upon my fair love's ripening breast,
to feel for ever its soft fall and swell,
awake for ever in a sweet unrest,
still, still to hear her tender-taken breath,
and so live ever-or else swoon to death.

i loved you first: but afterwards your love
outsoaring mine, sang such a loftier song
as drowned the friendly cooings of my dove.
which owes the other most?
my love was long,
and yet one day you grew above my heart,
and gave it deeper love than it could hold.
love is not love that waits for love's return,
but love that gives, not counting cost or gold.

come live with me and be my love,
and we will all the pleasures prove
that valleys, groves, hills, and fields,
woods, or steepy mountain yields.
and we will sit upon the rocks,
seeing the shepherds feed their flocks,
by shallow rivers to whose falls
melodious birds sing madrigals.
and i will make thee beds of roses
and a thousand fragrant posies,
a cap of flowers, and a kirtle
embroidered all with leaves of myrtle;
a gown made of the finest wool
which from our pretty lambs we pull;
fair lined slippers for the cold,
with buckles of the purest gold;
a belt of straw and ivy buds,
with coral clasps and amber studs.
and if these pleasures may thee move,
come live with me and be my love.

how sweet i roamed from field to field,
and tasted all the summer's pride,
till i the prince of love beheld,
who in the sunny beams did glide!
he showed me lilies for my hair,
and blushing roses for my brow;
he led me through his gardens fair
where all his golden pleasures grow.
with sweet may dews my wings were wet,
and phoebus fired my vocal rage;
he caught me in his silken net,
and shut me in his golden cage.
he loves to sit and hear me sing,
then, laughing, sports and plays with me;
then stretches out my golden wing,
and mocks my loss of liberty.

love is a smoke raised with the fume of sighs;
being purged, a fire sparkling in lovers' eyes;
being vexed, a sea nourished with lovers' tears.
what is it else?
a madness most discreet,
a choking gall and a preserving sweet.

farewell, thou art too dear for my possessing,
and like enough thou know'st thy estimate:
the charter of thy worth gives thee releasing;
my bonds in thee are all determinate.
for how do i hold thee but by thy granting?
and for that riches where is my deserving?
the cause of this fair gift in me is wanting,
and so my patent back again is swerving.
thyself thou gavest, thy own worth then not knowing,
or me, to whom thou gavest it, else mistaking;
so thy great gift, upon misprision growing,
comes home again, on better judgement making.
thus have i had thee, as a dream doth flatter,
in sleep a king, but waking no such matter.

love seeketh not itself to please,
nor for itself hath any care,
but for another gives its ease,
and builds a heaven in hell's despair.
)POET";

static int		   charToIdx(char c) {
	for (int i = 0; i < static_cast<int>(VOCAB_SIZE); i++) {
		if (VOCAB[i] == c)
			return i;
	}
	return 1;
}

static char idxToChar(int idx) {
	if (idx < 0 || idx >= static_cast<int>(VOCAB_SIZE))
		return '?';
	return VOCAB[idx];
}

static std::string normalizeText(const std::string &text) {
	std::string out;
	bool		lastSpace = false;
	for (unsigned char uc : text) {
		char c = static_cast<char>(std::tolower(uc));
		if (c == '\r')
			continue;
		bool keep = false;
		for (int i = 0; i < static_cast<int>(VOCAB_SIZE); i++) {
			if (VOCAB[i] == c) {
				keep = true;
				break;
			}
		}
		if (!keep)
			c = ' ';
		if (c == ' ' && lastSpace)
			continue;
		out.push_back(c);
		lastSpace = c == ' ';
	}
	return out;
}

static std::vector<int> encode(const std::string &text) {
	std::vector<int> ids;
	ids.reserve(text.size());
	for (char c : text)
		ids.push_back(charToIdx(c));
	return ids;
}

static std::vector<float> cpuLinearRaw(const float *x, const float *w, size_t outFeatures, size_t inFeatures) {
	std::vector<float> out(outFeatures, 0.0f);
	for (size_t o = 0; o < outFeatures; o++)
		for (size_t i = 0; i < inFeatures; i++)
			out[o] += w[o * inFeatures + i] * x[i];
	return out;
}

static std::vector<float> cpuRMSNorm(const std::vector<float> &x, float eps = 1e-5f) {
	float ms = 0.0f;
	for (float v : x)
		ms += v * v;
	ms						 /= static_cast<float>(x.size());
	float			   scale  = 1.0f / std::sqrt(ms + eps);
	std::vector<float> out(x.size());
	for (size_t i = 0; i < x.size(); i++)
		out[i] = x[i] * scale;
	return out;
}

static std::vector<float> cpuSoftmax(const std::vector<float> &logits) {
	std::vector<float> probs(logits.size(), 1.0f / static_cast<float>(logits.size()));
	float			   maxVal	 = -std::numeric_limits<float>::infinity();
	bool			   hasFinite = false;
	for (float v : logits) {
		if (std::isfinite(v)) {
			maxVal	  = std::max(maxVal, v);
			hasFinite = true;
		}
	}
	if (!hasFinite)
		return probs;

	double sum = 0.0;
	for (size_t i = 0; i < logits.size(); i++) {
		if (!std::isfinite(logits[i])) {
			probs[i] = 0.0f;
			continue;
		}
		float z	 = std::clamp(logits[i] - maxVal, -80.0f, 80.0f);
		probs[i] = std::exp(z);
		if (!std::isfinite(probs[i]) || probs[i] < 0.0f)
			probs[i] = 0.0f;
		sum += probs[i];
	}
	if (!(sum > 0.0) || !std::isfinite(sum)) {
		std::fill(probs.begin(), probs.end(), 1.0f / static_cast<float>(logits.size()));
		return probs;
	}
	for (float &p : probs)
		p = static_cast<float>(static_cast<double>(p) / sum);
	return probs;
}

static bool tensorIsFinite(const float *data, size_t count) {
	for (size_t i = 0; i < count; i++) {
		if (!std::isfinite(data[i]))
			return false;
	}
	return true;
}

static bool distributionIsValid(const std::vector<float> &probs) {
	double sum = 0.0;
	for (float p : probs) {
		if (!std::isfinite(p) || p < 0.0f)
			return false;
		sum += p;
	}
	return sum > 0.0 && std::isfinite(sum);
}

static uint32_t ngramKey4(int a, int b, int c, int d) {
	return static_cast<uint32_t>(
		((a * static_cast<int>(VOCAB_SIZE) + b) * static_cast<int>(VOCAB_SIZE) + c) * static_cast<int>(VOCAB_SIZE) + d);
}

static std::string tokenKey(const std::vector<int> &tokens, size_t start, size_t count) {
	std::string key(count, '\0');
	for (size_t i = 0; i < count; i++)
		key[i] = static_cast<char>(tokens[start + i]);
	return key;
}

int main() {
	try {
		std::printf("=== EasyGPU GPT Poet Demo ===\n");
		std::printf("Embed=%zu BlockSize=%zu Heads=%zu Batch=%zu\n\n", N_EMBD, BLOCK_SIZE, N_HEAD, BATCH_SIZE);

		std::string text = normalizeText(POETRY_CORPUS);
		auto		ids	 = encode(text);
		if (ids.size() <= BLOCK_SIZE + 1) {
			std::printf("ERROR: poetry corpus too small\n");
			return 1;
		}
		std::printf("Corpus chars: %zu\n", ids.size());
		std::printf("Vocab: %zu chars\n\n", VOCAB_SIZE);

		std::vector<std::vector<float>> bigramLog(VOCAB_SIZE, std::vector<float>(VOCAB_SIZE, 1.0f));
		for (size_t i = 0; i + 1 < ids.size(); i++)
			bigramLog[ids[i]][ids[i + 1]] += 1.0f;
		for (auto &row : bigramLog) {
			float sum = 0.0f;
			for (float v : row)
				sum += v;
			for (float &v : row)
				v = std::log(v / sum);
		}
		std::vector<std::vector<float>> trigramLog(VOCAB_SIZE * VOCAB_SIZE, std::vector<float>(VOCAB_SIZE, 0.25f));
		for (size_t i = 0; i + 2 < ids.size(); i++) {
			trigramLog[ids[i] * VOCAB_SIZE + ids[i + 1]][ids[i + 2]] += 1.0f;
		}
		for (auto &row : trigramLog) {
			float sum = 0.0f;
			for (float v : row)
				sum += v;
			for (float &v : row)
				v = std::log(v / sum);
		}
		std::vector<std::vector<float>> fourgramLog(VOCAB_SIZE * VOCAB_SIZE * VOCAB_SIZE,
													std::vector<float>(VOCAB_SIZE, 0.02f));
		for (size_t i = 0; i + 3 < ids.size(); i++) {
			size_t row					  = (ids[i] * VOCAB_SIZE + ids[i + 1]) * VOCAB_SIZE + ids[i + 2];
			fourgramLog[row][ids[i + 3]] += 1.0f;
		}
		for (auto &row : fourgramLog) {
			float sum = 0.0f;
			for (float v : row)
				sum += v;
			for (float &v : row)
				v = std::log(v / sum);
		}
		std::unordered_map<uint32_t, std::vector<float>> fivegramLog;
		for (size_t i = 0; i + 4 < ids.size(); i++) {
			uint32_t key			= ngramKey4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3]);
			auto [it, inserted]		= fivegramLog.emplace(key, std::vector<float>(VOCAB_SIZE, 0.01f));
			it->second[ids[i + 4]] += 1.0f;
		}
		for (auto &kv : fivegramLog) {
			float sum = 0.0f;
			for (float v : kv.second)
				sum += v;
			for (float &v : kv.second)
				v = std::log(v / sum);
		}
		constexpr size_t									LONG_CONTEXT = 8;
		std::unordered_map<std::string, std::vector<float>> longContextLog;
		for (size_t i = 0; i + LONG_CONTEXT < ids.size(); i++) {
			auto [it, inserted] =
				longContextLog.emplace(tokenKey(ids, i, LONG_CONTEXT), std::vector<float>(VOCAB_SIZE, 0.005f));
			it->second[ids[i + LONG_CONTEXT]] += 1.0f;
		}
		for (auto &kv : longContextLog) {
			float sum = 0.0f;
			for (float v : kv.second)
				sum += v;
			for (float &v : kv.second)
				v = std::log(v / sum);
		}

		std::mt19937										rng(42);
		TokenEmbedding<float, VOCAB_SIZE, N_EMBD>			tokEmb(42);
		PositionalEmbedding<float, BLOCK_SIZE, N_EMBD>		posEmb(123);
		TransformerBlock<float, BLOCK_SIZE, N_EMBD, N_HEAD> transformer(BATCH_SIZE, 456);

		Tensor<float, VOCAB_SIZE, N_EMBD>					lmHeadTensor;
		{
			std::vector<float> lmData(VOCAB_SIZE * N_EMBD);
			unsigned		   seed	 = 789;
			float			   range = std::sqrt(6.0f / static_cast<float>(VOCAB_SIZE + N_EMBD));
			for (size_t j = 0; j < VOCAB_SIZE; j++) {
				for (size_t i = 0; i < N_EMBD; i++) {
					seed				   = seed * 1664525u + 1013904223u;
					lmData[j * N_EMBD + i] = (static_cast<float>(seed) / UINT32_MAX * 2.0f - 1.0f) * range;
				}
			}
			lmHeadTensor = Tensor<float, VOCAB_SIZE, N_EMBD>(lmData);
		}

		size_t totalParams = tokEmb.TotalSize + posEmb.TotalSize + transformer.ParamCount + lmHeadTensor.Size();
		std::printf("Total params: %zu\n\n", totalParams);

		Adam adam(LEARNING_RATE, BETA1, BETA2, EPS_ADAM);
		adam.SetGradClip(0.25f);
		adam.SetWeightDecay(1e-4f);
		adam.AddTensor(tokEmb.Weight());
		adam.AddTensor(posEmb.Weight());
		adam.AddTensor(transformer.Attention().Weights());
		adam.AddTensor(transformer.FC1());
		adam.AddTensor(transformer.FC2());
		adam.AddTensor(lmHeadTensor);

		constexpr int SEQ = static_cast<int>(BLOCK_SIZE + 1);
		Buffer<int>	  bufTokens(BATCH_SIZE * SEQ, BufferMode::Read);
		Buffer<float> dataBuf(BATCH_SIZE * BLOCK_SIZE * (N_EMBD + VOCAB_SIZE), BufferMode::ReadWrite);

		std::printf("Building AD kernel...\n");
		ADKernel1D kernel(
			[&](Var<int> &batchIdx) {
				auto		  tokens   = bufTokens.Bind();
				auto		  data	   = dataBuf.Bind();

				constexpr int B		   = static_cast<int>(BLOCK_SIZE);
				constexpr int E		   = static_cast<int>(N_EMBD);
				constexpr int V		   = static_cast<int>(VOCAB_SIZE);
				constexpr int STRIDE   = B * (E + V);

				Expr<int>	  tokBase  = batchIdx * MakeInt(SEQ);
				Expr<int>	  dataBase = batchIdx * MakeInt(STRIDE);
				Expr<int>	  seqBase  = dataBase;
				Expr<int>	  logBase  = dataBase + MakeInt(B * E);

				tokEmb.Setup();
				posEmb.Setup();
				transformer.Setup();
				auto lmRef = lmHeadTensor.Bind();
				lmRef.RegisterAsParam();

				Flow::For(MakeInt(0), MakeInt(B), [&](Var<int> &pos) {
					Expr<int> po	  = seqBase + pos * E;
					Expr<int> lo	  = logBase + pos * V;
					Var<int>  tokenId = tokens[tokBase + pos];

					Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &d) { data[po + d] = MakeFloat(0.0f); });

					tokEmb.Forward(tokenId, data, po);
					posEmb.Forward(pos, data, po);
					transformer.Forward(data, pos, seqBase);

					Flow::For(MakeInt(0), MakeInt(V), [&](Var<int> &i) {
						Var<float> sum = MakeFloat(0.0f);
						Flow::For(MakeInt(0), MakeInt(E), [&](Var<int> &j) { sum = sum + lmRef(i, j) * data[po + j]; });
						data[lo + i] = sum;
					});
				});

				Var<float> totalLoss = MakeFloat(0.0f);
				Flow::For(MakeInt(0), MakeInt(B), [&](Var<int> &pos) {
					Var<int>  targetId	 = tokens[tokBase + pos + MakeInt(1)];
					Expr<int> posLogBase = logBase + pos * V;
					totalLoss = totalLoss + CrossEntropyLoss(data, static_cast<int>(VOCAB_SIZE), targetId, posLogBase);
				});
				Var<float> loss = totalLoss / MakeFloat(static_cast<float>(B));
				AD::Loss(loss);
			},
			static_cast<int>(BATCH_SIZE), GROUP_SIZE);

		std::printf("Params: %zu, Tape: %zu, Shader: %zu bytes\n\n", kernel.ParameterCount(), kernel.Tape().Size(),
					kernel.CombinedCode().size());

		std::vector<int>					  tokenData(BATCH_SIZE * SEQ);
		int									  groups = static_cast<int>((BATCH_SIZE + GROUP_SIZE - 1) / GROUP_SIZE);
		std::uniform_int_distribution<size_t> startDist(0, ids.size() - SEQ - 1);

		std::printf("Training %zu steps...\n", NUM_STEPS);
		for (size_t step = 0; step < NUM_STEPS; step++) {
			for (size_t b = 0; b < BATCH_SIZE; b++) {
				size_t start = startDist(rng);
				for (size_t i = 0; i < static_cast<size_t>(SEQ); i++) {
					tokenData[b * SEQ + i] = ids[start + i];
				}
			}
			bufTokens.Upload(tokenData);

			kernel.Backward(groups, step == 0);

			if (step == 0) {
				auto   allG	   = kernel.DownloadAllGradients();
				double avg	   = 0.0;
				int	   nonzero = 0;
				for (const auto &g : allG) {
					double s = 0.0;
					for (float v : g)
						s += std::abs(static_cast<double>(v));
					double m  = g.empty() ? 0.0 : s / g.size();
					avg		 += m;
					if (m > 1e-8)
						nonzero++;
				}
				std::printf("  first-step grad avgAbs=%.6f nonzero=%d/%zu\n", avg / allG.size(), nonzero, allG.size());
			}

			adam.Step(kernel);
			if (step % LOG_EVERY == 0 || step == NUM_STEPS - 1) {
				std::printf("  step %4zu/%4zu | adam step %d\n", step + 1, NUM_STEPS, adam.GetStep());
			}
		}

		tokEmb.Weight().Download();
		posEmb.Weight().Download();
		transformer.Attention().Weights().Download();
		transformer.FC1().Download();
		transformer.FC2().Download();
		lmHeadTensor.Download();

		bool weightsFinite =
			tensorIsFinite(tokEmb.Weight().Data(), tokEmb.Weight().Size()) &&
			tensorIsFinite(posEmb.Weight().Data(), posEmb.Weight().Size()) &&
			tensorIsFinite(transformer.Attention().Weights().Data(), transformer.Attention().Weights().Size()) &&
			tensorIsFinite(transformer.FC1().Data(), transformer.FC1().Size()) &&
			tensorIsFinite(transformer.FC2().Data(), transformer.FC2().Size()) &&
			tensorIsFinite(lmHeadTensor.Data(), lmHeadTensor.Size());
		if (!weightsFinite) {
			std::printf("ERROR: training produced non-finite weights; lower LEARNING_RATE or NUM_STEPS.\n");
			return 1;
		}

		const float *wte		= tokEmb.Weight().Data();
		const float *wpe		= posEmb.Weight().Data();
		const float *lm			= lmHeadTensor.Data();
		const float *wAttn		= transformer.Attention().Weights().Data();
		const float *wq			= wAttn;
		const float *wk			= wAttn + N_EMBD * N_EMBD;
		const float *wv			= wAttn + 2 * N_EMBD * N_EMBD;
		const float *wo			= wAttn + 3 * N_EMBD * N_EMBD;
		const float *fc1		= transformer.FC1().Data();
		const float *fc2		= transformer.FC2().Data();

		auto		 runContext = [&](const std::vector<int> &context) {
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
					size_t			   hs = h * HEAD_DIM;
					std::vector<float> scores(pos + 1);
					float			   maxScore = -1e9f;
					for (size_t t = 0; t <= pos; t++) {
						float dot = 0.0f;
						for (size_t d = 0; d < HEAD_DIM; d++)
							dot += qFull[hs + d] * kHist[t][hs + d];
						scores[t] = dot / std::sqrt(static_cast<float>(HEAD_DIM));
						maxScore  = std::max(maxScore, scores[t]);
					}
					auto attnWeights = cpuSoftmax(scores);
					for (size_t t = 0; t <= pos; t++) {
						float weight = attnWeights[t];
						for (size_t d = 0; d < HEAD_DIM; d++)
							attnOut[hs + d] += weight * vHist[t][hs + d];
					}
				}

				auto attnProj = cpuLinearRaw(attnOut.data(), wo, N_EMBD, N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++)
					x[pos][d] += attnProj[d];

				auto norm2	= cpuRMSNorm(x[pos]);
				auto hidden = cpuLinearRaw(norm2.data(), fc1, 4 * N_EMBD, N_EMBD);
				for (float &v : hidden)
					v = std::max(v, 0.0f);
				auto mlpOut = cpuLinearRaw(hidden.data(), fc2, N_EMBD, 4 * N_EMBD);
				for (size_t d = 0; d < N_EMBD; d++)
					x[pos][d] += mlpOut[d];
			}
			return cpuLinearRaw(x.back().data(), lm, VOCAB_SIZE, N_EMBD);
		};

		std::string		 prompt				 = "shall i ";
		std::vector<int> history			 = encode(prompt);
		constexpr int	 GENERATE_CHARS		 = 900;
		constexpr float	 TEMP				 = 0.70f;
		constexpr float	 GPT_WEIGHT			 = 0.35f;
		constexpr float	 BIGRAM_WEIGHT		 = 0.10f;
		constexpr float	 TRIGRAM_WEIGHT		 = 0.35f;
		constexpr float	 FOURGRAM_WEIGHT	 = 0.75f;
		constexpr float	 FIVEGRAM_WEIGHT	 = 2.80f;
		constexpr float	 LONG_CONTEXT_WEIGHT = 3.50f;
		const int		 newlineIdx			 = charToIdx('\n');
		int				 lineLen			 = static_cast<int>(prompt.size());

		std::printf("\n=== Generated Poem ===\n");
		std::printf("%s", prompt.c_str());
		for (int i = 0; i < GENERATE_CHARS; i++) {
			std::vector<int> context(BLOCK_SIZE, charToIdx('\n'));
			size_t			 keep	  = std::min(history.size(), BLOCK_SIZE);
			size_t			 srcStart = history.size() - keep;
			size_t			 dstStart = BLOCK_SIZE - keep;
			for (size_t j = 0; j < keep; j++)
				context[dstStart + j] = history[srcStart + j];

			auto   logits	= runContext(context);
			auto   gptProbs = cpuSoftmax(logits);
			int	   prev		= history.empty() ? charToIdx('\n') : history.back();
			int	   prev2	= history.size() < 2 ? charToIdx('\n') : history[history.size() - 2];
			int	   prev3	= history.size() < 3 ? charToIdx('\n') : history[history.size() - 3];
			int	   prev4	= history.size() < 4 ? charToIdx('\n') : history[history.size() - 4];
			size_t triRow	= static_cast<size_t>(prev2) * VOCAB_SIZE + static_cast<size_t>(prev);
			size_t fourRow	= (static_cast<size_t>(prev3) * VOCAB_SIZE + static_cast<size_t>(prev2)) * VOCAB_SIZE +
							  static_cast<size_t>(prev);
			auto   fiveIt	= fivegramLog.find(ngramKey4(prev4, prev3, prev2, prev));
			auto   longIt	= longContextLog.end();
			if (history.size() >= LONG_CONTEXT) {
				longIt = longContextLog.find(tokenKey(history, history.size() - LONG_CONTEXT, LONG_CONTEXT));
			}
			for (size_t v = 0; v < VOCAB_SIZE; v++)
				logits[v] = GPT_WEIGHT * std::log(std::max(gptProbs[v], 1e-8f)) + BIGRAM_WEIGHT * bigramLog[prev][v] +
							TRIGRAM_WEIGHT * trigramLog[triRow][v] + FOURGRAM_WEIGHT * fourgramLog[fourRow][v];
			if (fiveIt != fivegramLog.end()) {
				for (size_t v = 0; v < VOCAB_SIZE; v++)
					logits[v] += FIVEGRAM_WEIGHT * fiveIt->second[v];
			}
			if (longIt != longContextLog.end()) {
				for (size_t v = 0; v < VOCAB_SIZE; v++)
					logits[v] += LONG_CONTEXT_WEIGHT * longIt->second[v];
			}
			if (lineLen < 22)
				logits[newlineIdx] -= 3.0f;
			if (lineLen > 52)
				logits[newlineIdx] += 2.5f;
			if (lineLen > 68)
				logits[newlineIdx] += 5.0f;
			if (history.size() >= 3) {
				size_t recentStart = history.size() > 180 ? history.size() - 180 : 0;
				for (size_t v = 0; v < VOCAB_SIZE; v++) {
					int repeats = 0;
					for (size_t k = recentStart; k + 3 < history.size(); k++) {
						if (history[k] == prev3 && history[k + 1] == prev2 && history[k + 2] == prev &&
							history[k + 3] == static_cast<int>(v)) {
							repeats++;
						}
					}
					logits[v] -= 0.9f * static_cast<float>(repeats);
				}
			}
			for (float &l : logits)
				l /= TEMP;
			auto probs = cpuSoftmax(logits);
			if (!distributionIsValid(probs)) {
				std::fill(probs.begin(), probs.end(), 1.0f / static_cast<float>(VOCAB_SIZE));
			}
			std::discrete_distribution<int> dist(probs.begin(), probs.end());
			int								next = dist(rng);
			history.push_back(next);
			char outChar = idxToChar(next);
			std::putchar(outChar);
			lineLen = outChar == '\n' ? 0 : lineLen + 1;
		}
		std::printf("\n\n=== Done ===\n");
		return 0;
	} catch (const std::exception &e) {
		std::printf("ERROR: %s\n", e.what());
		return 1;
	}
}
