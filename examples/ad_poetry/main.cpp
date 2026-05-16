/**
 * @file main.cpp
 * @brief Character-level poetry generation with GPU self-attention and AD.
 *
 * Trains a small self-attention model (4D features) to predict the next
 * character from the previous 4. Each character gets a fixed 4D Fourier
 * embedding. Learnable position offsets and 4×4 Q/K/V projections give
 * the attention mechanism discriminative power.
 *
 * Architecture (86 trainable parameters):
 *   input (4 pos x 4D fixed embedding) + learnable 4D position offset
 *   -> Q/K/V proj (4x4 weight + 4 bias) -> 4x4 dot-product attention
 *   -> softmax -> weighted sum (4D per position)
 *   -> sum across positions -> output proj (4x2 + 2 bias) -> MSE vs target
 *
 * Generation: predict 2D vector -> find nearest character in embedding space.
 * Optimizer: RMSprop with weight decay.
 */

#include <GPU.h>
#include <AD/ADKernel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::AD;

// =============================================================================
// Character vocabulary
// =============================================================================
static const char* VOCAB = "abcdefghijklmnopqrstuvwxyz .,;'!?\n";
static const int   VOCAB_SIZE = 31;
static const int   SEQ_LEN    = 4;
static const int   FEAT_DIM   = 4;  // increased from 2
static const int   OUT_DIM    = 2;  // target is still 2D (sin/cos only)

static int charToIdx(char c) {
	c = std::tolower(static_cast<unsigned char>(c));
	for (int i = 0; i < VOCAB_SIZE; i++)
		if (VOCAB[i] == c) return i;
	return 1;
}

static char idxToChar(int idx) {
	if (idx < 0 || idx >= VOCAB_SIZE) return '?';
	return VOCAB[idx];
}

// 4D Fourier embedding: two frequencies on the unit circle
static void charEmbedding(int idx, float &f0, float &f1, float &f2, float &f3) {
	double a1 = idx * 2.0 * 3.141592653589793 / VOCAB_SIZE;
	double a2 = idx * 4.0 * 3.141592653589793 / VOCAB_SIZE;
	f0 = static_cast<float>(std::sin(a1));
	f1 = static_cast<float>(std::cos(a1));
	f2 = static_cast<float>(std::sin(a2));
	f3 = static_cast<float>(std::cos(a2));
}

// Nearest character in 4D embedding space
static int nearestChar(float f0, float f1, float f2, float f3) {
	int best = 0;
	double bestDist = 1e30;
	for (int i = 0; i < VOCAB_SIZE; i++) {
		float e0, e1, e2, e3;
		charEmbedding(i, e0, e1, e2, e3);
		double d = (f0-e0)*(f0-e0) + (f1-e1)*(f1-e1)
		         + (f2-e2)*(f2-e2) + (f3-e3)*(f3-e3);
		if (d < bestDist) { bestDist = d; best = i; }
	}
	return best;
}

// =============================================================================
// Poetry corpus
// =============================================================================
static const char* POETRY_CORPUS = R"(
the road not taken
two roads diverged in a yellow wood
and sorry i could not travel both
and be one traveler long i stood
and looked down one as far as i could
to where it bent in the undergrowth

then took the other as just as fair
and having perhaps the better claim
because it was grassy and wanted wear
though as for that the passing there
had worn them really about the same

and both that morning equally lay
in leaves no step had trodden black
oh i kept the first for another day
yet knowing how way leads on to way
i doubted if i should ever come back

i shall be telling this with a sigh
somewhere ages and ages hence
two roads diverged in a wood and i
i took the one less traveled by
and that has made all the difference

shall i compare thee to a summers day
thou art more lovely and more temperate
rough winds do shake the darling buds of may
and summers lease hath all too short a date

she walks in beauty like the night
of cloudless climes and starry skies
and all thats best of dark and bright
meet in her aspect and her eyes

hope is the thing with feathers
that perches in the soul
and sings the tune without the words
and never stops at all

i wandered lonely as a cloud
that floats on high oer vales and hills
when all at once i saw a crowd
a host of golden daffodils

because i could not stop for death
he kindly stopped for me
the carriage held but just ourselves
and immortality

do not go gentle into that good night
old age should burn and rave at close of day
rage rage against the dying of the light

how do i love thee let me count the ways
i love thee to the depth and breadth and height
my soul can reach when feeling out of sight

tyger tyger burning bright
in the forests of the night
what immortal hand or eye
could frame thy fearful symmetry

there was a young lady from lee
who sailed on the deep blue sea
she danced with the moon
and sang a sweet tune
as happy as one could be

the moon has a face like the clock in the hall
she shines on thieves on the garden wall
on streets and fields and harbour quays
and birdies asleep in the forks of the trees

i have a little shadow that goes in and out with me
and what can be the use of him is more than i can see
he is very very like me from the heels up to the head
and i can see him jump before me when i jump into my bed

roses are red violets are blue
sugar is sweet and so are you

the wind was a torrent of darkness among the gusty trees
the moon was a ghostly galleon tossed upon cloudy seas
the road was a ribbon of moonlight over the purple moor
and the highwayman came riding riding riding
up to the old inn door
)";

// =============================================================================
// Build training examples
// =============================================================================
struct Example {
	float input[SEQ_LEN * FEAT_DIM];
	float target[OUT_DIM];
};

static std::vector<Example> buildDataset() {
	std::vector<Example> dataset;
	std::string text;
	for (const char *p = POETRY_CORPUS; *p; p++) {
		char c = std::tolower(static_cast<unsigned char>(*p));
		for (int i = 0; i < VOCAB_SIZE; i++)
			if (VOCAB[i] == c) { text += c; break; }
	}
	for (size_t i = 0; i + SEQ_LEN < text.size(); i++) {
		Example ex;
		for (int pos = 0; pos < SEQ_LEN; pos++) {
			int ci = charToIdx(text[i + pos]);
			charEmbedding(ci,
				ex.input[pos * FEAT_DIM + 0], ex.input[pos * FEAT_DIM + 1],
				ex.input[pos * FEAT_DIM + 2], ex.input[pos * FEAT_DIM + 3]);
		}
		int ti = charToIdx(text[i + SEQ_LEN]);
		float t2, t3;
		charEmbedding(ti, ex.target[0], ex.target[1], t2, t3);
		dataset.push_back(ex);
	}
	return dataset;
}

// =============================================================================
// Parameter layout (86 total, flat array)
// =============================================================================
// Q: 4x4=16 weight + 4 bias = 20  [0..19]
// K: 4x4=16 weight + 4 bias = 20  [20..39]
// V: 4x4=16 weight + 4 bias = 20  [40..59]
// O: 4x2=8 weight + 2 bias = 10   [60..69]
// P: 4x4=16 position offsets      [70..85]
static constexpr int NQ   = 20;  // Q weights + biases
static constexpr int NK   = 20;  // K
static constexpr int NV   = 20;  // V
static constexpr int NO   = 10;  // Output projection
static constexpr int NP   = 16;  // Position offsets
static constexpr int NW   = NQ + NK + NV + NO + NP;  // 86

static constexpr int OFF_Q = 0;
static constexpr int OFF_K = OFF_Q + NQ;   // 20
static constexpr int OFF_V = OFF_K + NK;   // 40
static constexpr int OFF_O = OFF_V + NV;   // 60
static constexpr int OFF_P = OFF_O + NO;   // 70

// Weight[i][j] stored at i * D + j (row i, col j, output varies fastest)
static int wIdx(int off, int i, int j, int dim) { return off + i * dim + j; }

// =============================================================================
// CPU forward reference (must match GPU kernel exactly)
// =============================================================================
static void cpuAffine(const float* W, const float* bias, int dim,
                      const float* x, float* out) {
	for (int j = 0; j < dim; j++) {
		float s = bias[j];
		for (int i = 0; i < dim; i++) s += W[i * dim + j] * x[i];
		out[j] = s;
	}
}

static void cpuForward(const std::vector<float> &W,
                       const float* fin, float* fout) {
	// Add position offsets
	float x[4][4];
	for (int i = 0; i < 4; i++) {
		for (int d = 0; d < 4; d++)
			x[i][d] = fin[i * 4 + d] + W[OFF_P + i * 4 + d];
	}

	// Q, K, V projections
	float q[4][4], k[4][4], v[4][4];
	for (int i = 0; i < 4; i++) {
		cpuAffine(&W[OFF_Q], &W[OFF_Q + 16], 4, x[i], q[i]);
		cpuAffine(&W[OFF_K], &W[OFF_K + 16], 4, x[i], k[i]);
		cpuAffine(&W[OFF_V], &W[OFF_V + 16], 4, x[i], v[i]);
	}

	// Attention scores + softmax (scaled by 1/sqrt(d_k))
	float scale = 1.0f / std::sqrt(static_cast<float>(FEAT_DIM));
	float a[4][4];
	for (int i = 0; i < 4; i++) {
		float rowSum = 0;
		for (int j = 0; j < 4; j++) {
			float s = 0;
			for (int d = 0; d < 4; d++) s += q[i][d] * k[j][d];
			a[i][j] = std::exp(s * scale);
			rowSum += a[i][j];
		}
		for (int j = 0; j < 4; j++) a[i][j] /= rowSum;
	}

	// Weighted sum of V
	float y[4][4];
	for (int i = 0; i < 4; i++) {
		for (int d = 0; d < 4; d++) {
			y[i][d] = 0;
			for (int j = 0; j < 4; j++) y[i][d] += a[i][j] * v[j][d];
		}
	}

	// Sum across positions
	float sum[4] = {0,0,0,0};
	for (int i = 0; i < 4; i++)
		for (int d = 0; d < 4; d++)
			sum[d] += y[i][d];

	// Output projection: 4D -> 2D
	float out[2];
	for (int j = 0; j < 2; j++) {
		float s = W[OFF_O + 8 + j];  // bias
		for (int i = 0; i < 4; i++) s += W[OFF_O + i * 2 + j] * sum[i];
		out[j] = s;
	}
	fout[0] = out[0]; fout[1] = out[1];
}

static double computeLoss(const std::vector<float> &W,
                          const std::vector<float> &x_data,
                          const std::vector<float> &y_data, size_t N) {
	double total = 0;
	for (size_t i = 0; i < N; i++) {
		float out[2];
		cpuForward(W, &x_data[i * SEQ_LEN * FEAT_DIM], out);
		float d0 = out[0] - y_data[i * 2 + 0];
		float d1 = out[1] - y_data[i * 2 + 1];
		total += d0 * d0 + d1 * d1;
	}
	return total;
}

// =============================================================================
// Main
// =============================================================================
int main() {
	try {
		std::printf("=== GPU Poetry Transformer with AD ===\n\n");

		auto dataset = buildDataset();
		size_t N = dataset.size();
		std::printf("Corpus: %zu training examples (vocab=%d, feat=%dD)\n",
		            N, VOCAB_SIZE, FEAT_DIM);
		std::printf("Arch: %d-pos self-attention, %dD features, %d params\n",
		            SEQ_LEN, FEAT_DIM, NW);
		if (N < 100) { std::printf("ERROR: Not enough training data\n"); return 1; }

		constexpr int groupSize = 256;
		int groups = static_cast<int>((N + groupSize - 1) / groupSize);
		size_t N_padded = groups * groupSize;

		std::vector<float> x_data(N_padded * SEQ_LEN * FEAT_DIM, 0.0f);
		std::vector<float> y_data(N_padded * OUT_DIM, 0.0f);
		for (size_t i = 0; i < N; i++) {
			for (int j = 0; j < SEQ_LEN * FEAT_DIM; j++)
				x_data[i * SEQ_LEN * FEAT_DIM + j] = dataset[i].input[j];
			for (int j = 0; j < OUT_DIM; j++)
				y_data[i * OUT_DIM + j] = dataset[i].target[j];
		}

		// Xavier init
		std::vector<float> W_data(NW);
		std::mt19937 rng(42);
		auto xavier = [&](int off, int nW, int fanIn, int fanOut) {
			float s = std::sqrt(6.0f / (fanIn + fanOut));
			for (int i = 0; i < nW; i++)
				W_data[off + i] = std::uniform_real_distribution<float>(-s, s)(rng);
		};
		xavier(OFF_Q, 16, 4, 4);  // Q weights
		for (int i = 0; i < 4; i++) W_data[OFF_Q + 16 + i] = 0.0f; // Q bias=0
		xavier(OFF_K, 16, 4, 4);
		for (int i = 0; i < 4; i++) W_data[OFF_K + 16 + i] = 0.0f;
		xavier(OFF_V, 16, 4, 4);
		for (int i = 0; i < 4; i++) W_data[OFF_V + 16 + i] = 0.0f;
		xavier(OFF_O, 8, 4, 2);
		for (int i = 0; i < 2; i++) W_data[OFF_O + 8 + i] = 0.0f;
		for (int i = 0; i < NP; i++) W_data[OFF_P + i] = 0.0f; // pos offsets=0

		Buffer<float> buf_x(x_data, BufferMode::Read);
		Buffer<float> buf_y(y_data, BufferMode::Read);
		Buffer<float> buf_W(W_data, BufferMode::ReadWrite);

		// =====================================================================
		// AD Kernel — 4D self-attention transformer
		// =====================================================================
		ADKernel1D kernel([&](Var<int> &id) {
			auto x_ref = buf_x.Bind();
			auto y_ref = buf_y.Bind();
			auto W_ref = buf_W.Bind();

			// -- Input features (4 positions x 4D) --
			// Position 0
			auto f00 = x_ref[id*16 + 0]; auto f01 = x_ref[id*16 + 1];
			auto f02 = x_ref[id*16 + 2]; auto f03 = x_ref[id*16 + 3];
			// Position 1
			auto f10 = x_ref[id*16 + 4]; auto f11 = x_ref[id*16 + 5];
			auto f12 = x_ref[id*16 + 6]; auto f13 = x_ref[id*16 + 7];
			// Position 2
			auto f20 = x_ref[id*16 + 8]; auto f21 = x_ref[id*16 + 9];
			auto f22 = x_ref[id*16 +10]; auto f23 = x_ref[id*16 +11];
			// Position 3
			auto f30 = x_ref[id*16 +12]; auto f31 = x_ref[id*16 +13];
			auto f32 = x_ref[id*16 +14]; auto f33 = x_ref[id*16 +15];
			// Target
			auto t0 = y_ref[id*2 + 0]; auto t1 = y_ref[id*2 + 1];

			// -- Q weights [0..15] and biases [16..19] --
			auto q00=W_ref[ 0];auto q01=W_ref[ 1];auto q02=W_ref[ 2];auto q03=W_ref[ 3];
			auto q10=W_ref[ 4];auto q11=W_ref[ 5];auto q12=W_ref[ 6];auto q13=W_ref[ 7];
			auto q20=W_ref[ 8];auto q21=W_ref[ 9];auto q22=W_ref[10];auto q23=W_ref[11];
			auto q30=W_ref[12];auto q31=W_ref[13];auto q32=W_ref[14];auto q33=W_ref[15];
			auto qb0=W_ref[16];auto qb1=W_ref[17];auto qb2=W_ref[18];auto qb3=W_ref[19];

			// -- K weights [20..35] and biases [36..39] --
			auto k00=W_ref[20];auto k01=W_ref[21];auto k02=W_ref[22];auto k03=W_ref[23];
			auto k10=W_ref[24];auto k11=W_ref[25];auto k12=W_ref[26];auto k13=W_ref[27];
			auto k20=W_ref[28];auto k21=W_ref[29];auto k22=W_ref[30];auto k23=W_ref[31];
			auto k30=W_ref[32];auto k31=W_ref[33];auto k32=W_ref[34];auto k33=W_ref[35];
			auto kb0=W_ref[36];auto kb1=W_ref[37];auto kb2=W_ref[38];auto kb3=W_ref[39];

			// -- V weights [40..55] and biases [56..59] --
			auto v00=W_ref[40];auto v01=W_ref[41];auto v02=W_ref[42];auto v03=W_ref[43];
			auto v10=W_ref[44];auto v11=W_ref[45];auto v12=W_ref[46];auto v13=W_ref[47];
			auto v20=W_ref[48];auto v21=W_ref[49];auto v22=W_ref[50];auto v23=W_ref[51];
			auto v30=W_ref[52];auto v31=W_ref[53];auto v32=W_ref[54];auto v33=W_ref[55];
			auto vb0=W_ref[56];auto vb1=W_ref[57];auto vb2=W_ref[58];auto vb3=W_ref[59];

			// -- Output weights [60..67] and biases [68..69] --
			auto o00=W_ref[60];auto o01=W_ref[61];auto o10=W_ref[62];auto o11=W_ref[63];
			auto o20=W_ref[64];auto o21=W_ref[65];auto o30=W_ref[66];auto o31=W_ref[67];
			auto ob0=W_ref[68];auto ob1=W_ref[69];

			// -- Position offsets [70..85] --
			auto p00=W_ref[70];auto p01=W_ref[71];auto p02=W_ref[72];auto p03=W_ref[73];
			auto p10=W_ref[74];auto p11=W_ref[75];auto p12=W_ref[76];auto p13=W_ref[77];
			auto p20=W_ref[78];auto p21=W_ref[79];auto p22=W_ref[80];auto p23=W_ref[81];
			auto p30=W_ref[82];auto p31=W_ref[83];auto p32=W_ref[84];auto p33=W_ref[85];

			// === Add position offsets: x[pos][d] = f[pos][d] + P[pos][d] ===
			Var<float> x00,x01,x02,x03, x10,x11,x12,x13;
			Var<float> x20,x21,x22,x23, x30,x31,x32,x33;
			x00=f00+p00; x01=f01+p01; x02=f02+p02; x03=f03+p03;
			x10=f10+p10; x11=f11+p11; x12=f12+p12; x13=f13+p13;
			x20=f20+p20; x21=f21+p21; x22=f22+p22; x23=f23+p23;
			x30=f30+p30; x31=f31+p31; x32=f32+p32; x33=f33+p33;

			// === Helper: affine4 — 4D -> 4D projection ===
			// out_j = sum_i w_ij * x_i + b_j
			auto affine4 = [](auto &x0,auto &x1,auto &x2,auto &x3,
			                  auto &w00,auto &w01,auto &w02,auto &w03,
			                  auto &w10,auto &w11,auto &w12,auto &w13,
			                  auto &w20,auto &w21,auto &w22,auto &w23,
			                  auto &w30,auto &w31,auto &w32,auto &w33,
			                  auto &b0, auto &b1, auto &b2, auto &b3,
			                  Var<float> &o0, Var<float> &o1,
			                  Var<float> &o2, Var<float> &o3) {
				// j=0
				Var<float> m00,m10,m20,m30, s01_0,s23_0,s0, sb0;
				m00=w00*x0; m10=w10*x1; m20=w20*x2; m30=w30*x3;
				s01_0=m00+m10; s23_0=m20+m30; s0=s01_0+s23_0; sb0=s0+b0; o0=sb0;
				// j=1
				Var<float> m01,m11,m21,m31, s01_1,s23_1,s1, sb1;
				m01=w01*x0; m11=w11*x1; m21=w21*x2; m31=w31*x3;
				s01_1=m01+m11; s23_1=m21+m31; s1=s01_1+s23_1; sb1=s1+b1; o1=sb1;
				// j=2
				Var<float> m02,m12,m22,m32, s01_2,s23_2,s2, sb2;
				m02=w02*x0; m12=w12*x1; m22=w22*x2; m32=w32*x3;
				s01_2=m02+m12; s23_2=m22+m32; s2=s01_2+s23_2; sb2=s2+b2; o2=sb2;
				// j=3
				Var<float> m03,m13,m23,m33, s01_3,s23_3,s3, sb3;
				m03=w03*x0; m13=w13*x1; m23=w23*x2; m33=w33*x3;
				s01_3=m03+m13; s23_3=m23+m33; s3=s01_3+s23_3; sb3=s3+b3; o3=sb3;
			};

			// === Q, K, V projections for all 4 positions ===
			Var<float> q00v,q01v,q02v,q03v, q10v,q11v,q12v,q13v;
			Var<float> q20v,q21v,q22v,q23v, q30v,q31v,q32v,q33v;
			Var<float> k00v,k01v,k02v,k03v, k10v,k11v,k12v,k13v;
			Var<float> k20v,k21v,k22v,k23v, k30v,k31v,k32v,k33v;
			Var<float> v00v,v01v,v02v,v03v, v10v,v11v,v12v,v13v;
			Var<float> v20v,v21v,v22v,v23v, v30v,v31v,v32v,v33v;

			affine4(x00,x01,x02,x03, q00,q01,q02,q03,q10,q11,q12,q13,q20,q21,q22,q23,q30,q31,q32,q33, qb0,qb1,qb2,qb3, q00v,q01v,q02v,q03v);
			affine4(x00,x01,x02,x03, k00,k01,k02,k03,k10,k11,k12,k13,k20,k21,k22,k23,k30,k31,k32,k33, kb0,kb1,kb2,kb3, k00v,k01v,k02v,k03v);
			affine4(x00,x01,x02,x03, v00,v01,v02,v03,v10,v11,v12,v13,v20,v21,v22,v23,v30,v31,v32,v33, vb0,vb1,vb2,vb3, v00v,v01v,v02v,v03v);
			affine4(x10,x11,x12,x13, q00,q01,q02,q03,q10,q11,q12,q13,q20,q21,q22,q23,q30,q31,q32,q33, qb0,qb1,qb2,qb3, q10v,q11v,q12v,q13v);
			affine4(x10,x11,x12,x13, k00,k01,k02,k03,k10,k11,k12,k13,k20,k21,k22,k23,k30,k31,k32,k33, kb0,kb1,kb2,kb3, k10v,k11v,k12v,k13v);
			affine4(x10,x11,x12,x13, v00,v01,v02,v03,v10,v11,v12,v13,v20,v21,v22,v23,v30,v31,v32,v33, vb0,vb1,vb2,vb3, v10v,v11v,v12v,v13v);
			affine4(x20,x21,x22,x23, q00,q01,q02,q03,q10,q11,q12,q13,q20,q21,q22,q23,q30,q31,q32,q33, qb0,qb1,qb2,qb3, q20v,q21v,q22v,q23v);
			affine4(x20,x21,x22,x23, k00,k01,k02,k03,k10,k11,k12,k13,k20,k21,k22,k23,k30,k31,k32,k33, kb0,kb1,kb2,kb3, k20v,k21v,k22v,k23v);
			affine4(x20,x21,x22,x23, v00,v01,v02,v03,v10,v11,v12,v13,v20,v21,v22,v23,v30,v31,v32,v33, vb0,vb1,vb2,vb3, v20v,v21v,v22v,v23v);
			affine4(x30,x31,x32,x33, q00,q01,q02,q03,q10,q11,q12,q13,q20,q21,q22,q23,q30,q31,q32,q33, qb0,qb1,qb2,qb3, q30v,q31v,q32v,q33v);
			affine4(x30,x31,x32,x33, k00,k01,k02,k03,k10,k11,k12,k13,k20,k21,k22,k23,k30,k31,k32,k33, kb0,kb1,kb2,kb3, k30v,k31v,k32v,k33v);
			affine4(x30,x31,x32,x33, v00,v01,v02,v03,v10,v11,v12,v13,v20,v21,v22,v23,v30,v31,v32,v33, vb0,vb1,vb2,vb3, v30v,v31v,v32v,v33v);

			// === Accessor helpers for Q, K, V by position ===
			auto qv = [&](int i, int d) -> Var<float>& {
				if (i==0) return (d==0?q00v:d==1?q01v:d==2?q02v:q03v);
				if (i==1) return (d==0?q10v:d==1?q11v:d==2?q12v:q13v);
				if (i==2) return (d==0?q20v:d==1?q21v:d==2?q22v:q23v);
				return (d==0?q30v:d==1?q31v:d==2?q32v:q33v);
			};
			auto kv = [&](int j, int d) -> Var<float>& {
				if (j==0) return (d==0?k00v:d==1?k01v:d==2?k02v:k03v);
				if (j==1) return (d==0?k10v:d==1?k11v:d==2?k12v:k13v);
				if (j==2) return (d==0?k20v:d==1?k21v:d==2?k22v:k23v);
				return (d==0?k30v:d==1?k31v:d==2?k32v:k33v);
			};
			auto vv = [&](int j, int d) -> Var<float>& {
				if (j==0) return (d==0?v00v:d==1?v01v:d==2?v02v:v03v);
				if (j==1) return (d==0?v10v:d==1?v11v:d==2?v12v:v13v);
				if (j==2) return (d==0?v20v:d==1?v21v:d==2?v22v:v23v);
				return (d==0?v30v:d==1?v31v:d==2?v32v:v33v);
			};

			// === Helper: dot4 — 4D dot product ===
			auto dot4 = [](auto &a0,auto &a1,auto &a2,auto &a3,
			               auto &b0,auto &b1,auto &b2,auto &b3,
			               Var<float> &out) {
				Var<float> m0,m1,m2,m3, s01,s23,s;
				m0=a0*b0; m1=a1*b1; m2=a2*b2; m3=a3*b3;
				s01=m0+m1; s23=m2+m3; s=s01+s23; out=s;
			};

			// === Attention scores: s_ij = qi · kj (4x4 = 16 dot products) ===
			Var<float> s00,s01,s02,s03, s10,s11,s12,s13;
			Var<float> s20,s21,s22,s23, s30,s31,s32,s33;
			dot4(qv(0,0),qv(0,1),qv(0,2),qv(0,3), kv(0,0),kv(0,1),kv(0,2),kv(0,3), s00);
			dot4(qv(0,0),qv(0,1),qv(0,2),qv(0,3), kv(1,0),kv(1,1),kv(1,2),kv(1,3), s01);
			dot4(qv(0,0),qv(0,1),qv(0,2),qv(0,3), kv(2,0),kv(2,1),kv(2,2),kv(2,3), s02);
			dot4(qv(0,0),qv(0,1),qv(0,2),qv(0,3), kv(3,0),kv(3,1),kv(3,2),kv(3,3), s03);
			dot4(qv(1,0),qv(1,1),qv(1,2),qv(1,3), kv(0,0),kv(0,1),kv(0,2),kv(0,3), s10);
			dot4(qv(1,0),qv(1,1),qv(1,2),qv(1,3), kv(1,0),kv(1,1),kv(1,2),kv(1,3), s11);
			dot4(qv(1,0),qv(1,1),qv(1,2),qv(1,3), kv(2,0),kv(2,1),kv(2,2),kv(2,3), s12);
			dot4(qv(1,0),qv(1,1),qv(1,2),qv(1,3), kv(3,0),kv(3,1),kv(3,2),kv(3,3), s13);
			dot4(qv(2,0),qv(2,1),qv(2,2),qv(2,3), kv(0,0),kv(0,1),kv(0,2),kv(0,3), s20);
			dot4(qv(2,0),qv(2,1),qv(2,2),qv(2,3), kv(1,0),kv(1,1),kv(1,2),kv(1,3), s21);
			dot4(qv(2,0),qv(2,1),qv(2,2),qv(2,3), kv(2,0),kv(2,1),kv(2,2),kv(2,3), s22);
			dot4(qv(2,0),qv(2,1),qv(2,2),qv(2,3), kv(3,0),kv(3,1),kv(3,2),kv(3,3), s23);
			dot4(qv(3,0),qv(3,1),qv(3,2),qv(3,3), kv(0,0),kv(0,1),kv(0,2),kv(0,3), s30);
			dot4(qv(3,0),qv(3,1),qv(3,2),qv(3,3), kv(1,0),kv(1,1),kv(1,2),kv(1,3), s31);
			dot4(qv(3,0),qv(3,1),qv(3,2),qv(3,3), kv(2,0),kv(2,1),kv(2,2),kv(2,3), s32);
			dot4(qv(3,0),qv(3,1),qv(3,2),qv(3,3), kv(3,0),kv(3,1),kv(3,2),kv(3,3), s33);

			// === Softmax per row (with scaled dot-product: divide by sqrt(d_k)) ===
			// This prevents exp overflow when weights drift during training.
			auto softmax4 = [](auto &s0,auto &s1,auto &s2,auto &s3,
			                   Var<float> &a0,Var<float> &a1,
			                   Var<float> &a2,Var<float> &a3) {
				Var<float> t0,t1,t2,t3, e0,e1,e2,e3, sum01,sum012,esum;
				float scale = 1.0f / std::sqrt(static_cast<float>(FEAT_DIM));
				t0=s0*MakeFloat(scale); t1=s1*MakeFloat(scale);
				t2=s2*MakeFloat(scale); t3=s3*MakeFloat(scale);
				e0=Exp(t0); e1=Exp(t1); e2=Exp(t2); e3=Exp(t3);
				sum01=e0+e1; sum012=sum01+e2; esum=sum012+e3;
				a0=e0/esum; a1=e1/esum; a2=e2/esum; a3=e3/esum;
			};

			Var<float> a00,a01,a02,a03, a10,a11,a12,a13;
			Var<float> a20,a21,a22,a23, a30,a31,a32,a33;
			softmax4(s00,s01,s02,s03, a00,a01,a02,a03);
			softmax4(s10,s11,s12,s13, a10,a11,a12,a13);
			softmax4(s20,s21,s22,s23, a20,a21,a22,a23);
			softmax4(s30,s31,s32,s33, a30,a31,a32,a33);

			// === Weighted sum: y_i_d = sum_j a_ij * V_j_d ===
			auto wsum = [](auto &a0,auto &a1,auto &a2,auto &a3,
			               auto &v0,auto &v1,auto &v2,auto &v3,
			               Var<float> &y) {
				Var<float> p0,p1,p2,p3, s01,s012;
				p0=a0*v0; p1=a1*v1; p2=a2*v2; p3=a3*v3;
				s01=p0+p1; s012=s01+p2; y=s012+p3;
			};

			// 4 positions x 4D = 16 weighted sums
			Var<float> y00,y01,y02,y03, y10,y11,y12,y13;
			Var<float> y20,y21,y22,y23, y30,y31,y32,y33;
			wsum(a00,a01,a02,a03, vv(0,0),vv(1,0),vv(2,0),vv(3,0), y00);
			wsum(a00,a01,a02,a03, vv(0,1),vv(1,1),vv(2,1),vv(3,1), y01);
			wsum(a00,a01,a02,a03, vv(0,2),vv(1,2),vv(2,2),vv(3,2), y02);
			wsum(a00,a01,a02,a03, vv(0,3),vv(1,3),vv(2,3),vv(3,3), y03);
			wsum(a10,a11,a12,a13, vv(0,0),vv(1,0),vv(2,0),vv(3,0), y10);
			wsum(a10,a11,a12,a13, vv(0,1),vv(1,1),vv(2,1),vv(3,1), y11);
			wsum(a10,a11,a12,a13, vv(0,2),vv(1,2),vv(2,2),vv(3,2), y12);
			wsum(a10,a11,a12,a13, vv(0,3),vv(1,3),vv(2,3),vv(3,3), y13);
			wsum(a20,a21,a22,a23, vv(0,0),vv(1,0),vv(2,0),vv(3,0), y20);
			wsum(a20,a21,a22,a23, vv(0,1),vv(1,1),vv(2,1),vv(3,1), y21);
			wsum(a20,a21,a22,a23, vv(0,2),vv(1,2),vv(2,2),vv(3,2), y22);
			wsum(a20,a21,a22,a23, vv(0,3),vv(1,3),vv(2,3),vv(3,3), y23);
			wsum(a30,a31,a32,a33, vv(0,0),vv(1,0),vv(2,0),vv(3,0), y30);
			wsum(a30,a31,a32,a33, vv(0,1),vv(1,1),vv(2,1),vv(3,1), y31);
			wsum(a30,a31,a32,a33, vv(0,2),vv(1,2),vv(2,2),vv(3,2), y32);
			wsum(a30,a31,a32,a33, vv(0,3),vv(1,3),vv(2,3),vv(3,3), y33);

			// === Sum across positions (4D) ===
			Var<float> sum0, sum1, sum2, sum3;
			{
				Var<float> s01_0,s012_0, s01_1,s012_1, s01_2,s012_2, s01_3,s012_3;
				s01_0=y00+y10; s012_0=s01_0+y20; sum0=s012_0+y30;
				s01_1=y01+y11; s012_1=s01_1+y21; sum1=s012_1+y31;
				s01_2=y02+y12; s012_2=s01_2+y22; sum2=s012_2+y32;
				s01_3=y03+y13; s012_3=s01_3+y23; sum3=s012_3+y33;
			}

			// === Output projection: 4D -> 2D ===
			Var<float> out0, out1;
			{
				// out0 = o00*sum0 + o10*sum1 + o20*sum2 + o30*sum3 + ob0
				Var<float> m00,m10,m20,m30, s01,s23,s, sb;
				m00=o00*sum0; m10=o10*sum1; m20=o20*sum2; m30=o30*sum3;
				s01=m00+m10; s23=m20+m30; s=s01+s23; sb=s+ob0; out0=sb;
			}
			{
				Var<float> m01,m11,m21,m31, s01,s23,s, sb;
				m01=o01*sum0; m11=o11*sum1; m21=o21*sum2; m31=o31*sum3;
				s01=m01+m11; s23=m21+m31; s=s01+s23; sb=s+ob1; out1=sb;
			}

			// === MSE loss ===
			Var<float> diff0, diff1, loss0, loss1, loss;
			diff0 = out0 - t0;  loss0 = diff0 * diff0;
			diff1 = out1 - t1;  loss1 = diff1 * diff1;
			loss = loss0 + loss1;

			// === Register all 86 parameters ===
			// Q
			AD::Param(q00);AD::Param(q01);AD::Param(q02);AD::Param(q03);
			AD::Param(q10);AD::Param(q11);AD::Param(q12);AD::Param(q13);
			AD::Param(q20);AD::Param(q21);AD::Param(q22);AD::Param(q23);
			AD::Param(q30);AD::Param(q31);AD::Param(q32);AD::Param(q33);
			AD::Param(qb0);AD::Param(qb1);AD::Param(qb2);AD::Param(qb3);
			// K
			AD::Param(k00);AD::Param(k01);AD::Param(k02);AD::Param(k03);
			AD::Param(k10);AD::Param(k11);AD::Param(k12);AD::Param(k13);
			AD::Param(k20);AD::Param(k21);AD::Param(k22);AD::Param(k23);
			AD::Param(k30);AD::Param(k31);AD::Param(k32);AD::Param(k33);
			AD::Param(kb0);AD::Param(kb1);AD::Param(kb2);AD::Param(kb3);
			// V
			AD::Param(v00);AD::Param(v01);AD::Param(v02);AD::Param(v03);
			AD::Param(v10);AD::Param(v11);AD::Param(v12);AD::Param(v13);
			AD::Param(v20);AD::Param(v21);AD::Param(v22);AD::Param(v23);
			AD::Param(v30);AD::Param(v31);AD::Param(v32);AD::Param(v33);
			AD::Param(vb0);AD::Param(vb1);AD::Param(vb2);AD::Param(vb3);
			// Output
			AD::Param(o00);AD::Param(o01);AD::Param(o10);AD::Param(o11);
			AD::Param(o20);AD::Param(o21);AD::Param(o30);AD::Param(o31);
			AD::Param(ob0);AD::Param(ob1);
			// Position offsets
			AD::Param(p00);AD::Param(p01);AD::Param(p02);AD::Param(p03);
			AD::Param(p10);AD::Param(p11);AD::Param(p12);AD::Param(p13);
			AD::Param(p20);AD::Param(p21);AD::Param(p22);AD::Param(p23);
			AD::Param(p30);AD::Param(p31);AD::Param(p32);AD::Param(p33);
			AD::Loss(loss);
		}, N_padded, groupSize);

		std::printf("Parameters: %zu, Tape entries: %zu\n",
		            kernel.ParameterCount(), kernel.Tape().Size());

		// =====================================================================
		// Compile
		// =====================================================================
		std::printf("=== Compiling combined pipeline ===\n");
		kernel.Backward(groups, true);
		std::printf("Combined pipeline OK!\n\n");

		// =====================================================================
		// Gradient verification (first 8 params)
		// =====================================================================
		std::printf("=== Gradient Verification (first 8 of %d params) ===\n", NW);
		auto sumGrad = [&](int pidx) {
			auto g = kernel.Gradient(pidx);
			double s = 0;
			for (size_t i = 0; i < N_padded; i++) s += g[i];
			return s;
		};
		float eps = 5e-4f;
		const char* pname[] = {"Q00","Q01","Q02","Q03","Q10","Q11","Q12","Q13"};
		for (int p = 0; p < 8; p++) {
			auto Wp = W_data, Wm = W_data;
			Wp[p] += eps; Wm[p] -= eps;
			double fd = (computeLoss(Wp, x_data, y_data, N) -
			             computeLoss(Wm, x_data, y_data, N)) / (2.0 * eps);
			double ad = sumGrad(p);
			double err = std::abs(fd - ad);
			double denom = std::max(std::abs(fd), 1e-6);
			std::printf("  %s: AD=%.6f FD=%.6f rel_err=%.2e %s\n",
			            pname[p], ad, fd, err/denom,
			            (err/denom < 0.10 || err < 1e-3) ? "OK" : "WARN");
		}
		std::printf("  ... (%d remaining params omitted)\n\n", NW - 8);

		// =====================================================================
		// Training — RMSprop with weight decay
		// =====================================================================
		constexpr int trainSteps = 40000;
		float lr = 0.00005f;
		float beta = 0.9f;
		float weightDecay = 0.0001f;
		float gradClip = 0.1f;

		std::printf("=== Training (%d RMSprop steps, lr=%.5f, decay=%.4f, clip=%.2f) ===\n",
		            trainSteps, lr, weightDecay, gradClip);

		std::vector<float> sq_avg(NW, 0.0f);

		for (int step = 0; step < trainSteps; step++) {
			kernel.Backward(groups, true);

			for (int p = 0; p < NW; p++) {
				auto grad = kernel.Gradient(p);
				double tg = 0;
				for (size_t i = 0; i < N_padded; i++) tg += grad[i];
				double g = tg / N;
				// Clip per-example-mean gradient
				if (g >  gradClip) g =  gradClip;
				if (g < -gradClip) g = -gradClip;
				g += 2.0 * weightDecay * W_data[p];

				sq_avg[p] = beta * sq_avg[p] + (1.0f - beta) * (float)(g * g);
				float rms = std::sqrt(sq_avg[p] + 1e-8f);
				W_data[p] -= lr * g / rms;
			}
			buf_W.Upload(W_data);

			if (step % 500 == 0 || (step < 1000 && step % 100 == 0)) {
				double loss = computeLoss(W_data, x_data, y_data, N) / N;
				if (!std::isfinite(loss)) {
					std::printf("  Step %3d: loss=%.6f — DIVERGED, stopping\n", step, loss);
					break;
				}
				std::printf("  Step %3d: loss=%.6f\n", step, loss);
			}
		}
		double final_loss = computeLoss(W_data, x_data, y_data, N) / N;
		std::printf("\n=== Final Results ===\n  Final loss: %.6f\n", final_loss);

		// Accuracy — match output 2D against embedding 2D (first two dims)
		int correct = 0;
		for (size_t i = 0; i < N; i++) {
			float out[2];
			cpuForward(W_data, &x_data[i * SEQ_LEN * FEAT_DIM], out);
			// True char from y_data target (use first 2 embedding dims)
			int trueChar = 0;
			float bestD = 1e30;
			for (int c = 0; c < VOCAB_SIZE; c++) {
				float e0,e1,e2,e3;
				charEmbedding(c, e0,e1,e2,e3);
				float d = (y_data[i*2]-e0)*(y_data[i*2]-e0)
				        + (y_data[i*2+1]-e1)*(y_data[i*2+1]-e1);
				if (d < bestD) { bestD = d; trueChar = c; }
			}
			// Predicted char from 2D output
			int predChar = 0;
			bestD = 1e30;
			for (int c = 0; c < VOCAB_SIZE; c++) {
				float e0,e1,e2,e3;
				charEmbedding(c, e0,e1,e2,e3);
				float d = (out[0]-e0)*(out[0]-e0) + (out[1]-e1)*(out[1]-e1);
				if (d < bestD) { bestD = d; predChar = c; }
			}
			if (predChar == trueChar) correct++;
		}
		std::printf("  Next-char accuracy: %d/%zu (%.1f%%)\n", correct, N, 100.0f*correct/N);

		// =====================================================================
		// Generation
		// =====================================================================
		std::printf("\n=== Poetry Generation ===\n");
		const char* seeds[] = {
			"the ", "and ", "but ", "love", "dark", "moon", "wind", "when"
		};
		for (int s = 0; s < 8; s++) {
			std::string seq = seeds[s];
			while (seq.size() < 4) seq += ' ';
			seq = seq.substr(0, 4);
			std::printf("  \"%s\" -> \"", seq.c_str());
			for (int g = 0; g < 40; g++) {
				float fin[SEQ_LEN * FEAT_DIM], fout[2];
				for (int p = 0; p < 4; p++) {
					int ci = charToIdx(seq[seq.size() - 4 + p]);
					float e0,e1,e2,e3;
					charEmbedding(ci, e0,e1,e2,e3);
					fin[p*4+0]=e0; fin[p*4+1]=e1; fin[p*4+2]=e2; fin[p*4+3]=e3;
				}
				cpuForward(W_data, fin, fout);
				// Find nearest character (matching only first 2 embedding dims)
				int bestC = 0;
				float bestD = 1e30;
				for (int c = 0; c < VOCAB_SIZE; c++) {
					float e0,e1,e2,e3;
					charEmbedding(c, e0,e1,e2,e3);
					float d = (fout[0]-e0)*(fout[0]-e0) + (fout[1]-e1)*(fout[1]-e1);
					if (d < bestD) { bestD = d; bestC = c; }
				}
				seq += idxToChar(bestC);
			}
			std::printf("%s\"\n", seq.c_str() + 4);
		}
		std::printf("\n*** DEMO COMPLETE ***\n");
		return 0;

	} catch (const std::exception &e) {
		std::printf("EXCEPTION: %s\n", e.what());
		return 1;
	} catch (...) {
		std::printf("UNKNOWN EXCEPTION\n");
		return 1;
	}
}
