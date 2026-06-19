/**
 * @file main.cpp
 * @brief Show the Vulkan SPIR-V optimization inspection API with real shader output.
 */

#include <GPU.h>

#include <iostream>
#include <string>
#include <vector>

int main() {
	std::vector<Vec3> hdr(64, Vec3(1.0f, 0.5f, 0.25f));
	Buffer<Vec3>		 hdrInput(hdr);
	Buffer<Vec3>		 ldrOutput(hdr.size());

	Kernel1D			 kernel(
		"ToneMapInspection",
		[&](Int i) {
			auto src = hdrInput.Bind();
			auto dst = ldrOutput.Bind();

			Float3 color		= src[i];
			Float  exposure		= MakeFloat(1.25f);
			Float3 whiteBalance = MakeFloat3(1.03f, 0.98f, 0.92f);
			Float3 balanced		= color * whiteBalance;
			Float3 exposed		= balanced * exposure;

			Float  lumaA		= Dot(exposed, MakeFloat3(0.2126f, 0.7152f, 0.0722f));
			Float  lumaB		= Dot(exposed, MakeFloat3(0.2126f, 0.7152f, 0.0722f));
			Float3 acesTop		= exposed * (exposed * 2.51f + MakeFloat(0.03f));
			Float3 acesBottom	= exposed * (exposed * 2.43f + MakeFloat(0.59f)) + MakeFloat(0.14f);
			Float3 acesMapped	= Clamp(acesTop / acesBottom, 0.0f, 1.0f);
			Float  vignette		= Clamp(1.0f - Abs(lumaA - 0.5f) * 0.08f, 0.92f, 1.0f);
			Float3 normalized	= Normalize(exposed + 0.001f);
			Float  blend		= Clamp(lumaA / (1.0f + lumaA), 0.0f, 1.0f) * 0.15f;
			Float  dead			= (lumaB * 0.0f) + Dot(normalized, normalized) * 0.0f;

			If(MakeBool(false), [&] { dst[i] = MakeFloat3(dead, dead, dead); }).Else([&] {
				Float3 graded = Mix(acesMapped, normalized, blend) * vignette;
				dst[i]		  = Clamp(graded, 0.0f, 1.0f);
			});
		},
		256);

	std::cout << "=== Generated GLSL from GetCode() ===\n";
	std::cout << kernel.GetCode() << "\n";
	std::cout << "=== Optimized GLSL from GetOptimizedGLSL() ===\n";
	std::cout << kernel.GetOptimizedGLSL() << "\n";

	return 0;
}
