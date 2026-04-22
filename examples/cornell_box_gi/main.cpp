/**
 * CornellBoxGI:
 *      @Description    :   Real-time progressive path tracing for Cornell Box with NEE
 *      @Author         :   Assistant
 *      @Date           :   4/22/2026
 *
 *  A real-time global illumination demo using EasyGPU's compute kernel pipeline.
 *  Features:
 *      - 1spp per frame with temporal accumulation via HDR buffer
 *      - Next Event Estimation (NEE) for fast direct-light convergence
 *      - Progressive rendering via AppWindow + TexturePresenter
 */

#include <GPU.h>

#include <cmath>
#include <iostream>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;
using namespace GPU::Callables;

// =============================================================================
// Configuration
// =============================================================================
constexpr int WIDTH       = 512;
constexpr int HEIGHT      = 512;
constexpr int MAX_DEPTH   = 4;
constexpr int NUM_OBJECTS = 8; // Number of AABB objects in the scene

// =============================================================================
// GPU Struct Definitions
// =============================================================================
EASYGPU_STRUCT(Ray, (Vec3, origin), (Vec3, dir));

EASYGPU_STRUCT(Material, (Vec3, albedo), (int, type) // 0=diffuse, 1=metal, 2=light
);

EASYGPU_STRUCT(HitRec, (Vec3, p), (Vec3, normal), (float, t), (Material, mat));

// =============================================================================
// Random Number Generation
// =============================================================================
Callable<Float(Int &)> Random = [](Int &state) {
	state	   = (state * 747796405 + 2891336453) & 0x7FFFFFFF;
	Int word   = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
	Int result = (word >> 22) ^ word;
	result	   = Abs(result);

	Return(ToFloat(result) / 2147483647.0f);
};

Callable<Float3(Int &)> RandomInUnitSphere = [](Int &state) {
	Float3 p;
	For(0, 50, [&](Int &) {
		p = MakeFloat3(Random(state), Random(state), Random(state)) * 2.0f - MakeFloat3(1.0f, 1.0f, 1.0f);
	});

	Return(p);
};

Callable<Float3(Int &)> RandomUnitVector = [](Int &state) { Return(Normalize(RandomInUnitSphere(state))); };

// =============================================================================
// Ray Helpers
// =============================================================================
Callable<Float3(Var<Ray> &, Float)> RayAt = [](Var<Ray> &r, Float t) { Return(r.origin() + r.dir() * t); };

Callable<void(Var<Ray> &, Float3 &, Float3 &)> SetRay = [](Var<Ray> &r, Float3 &o, Float3 &d) {
	r.origin() = o;
	r.dir()	   = d;
};

// =============================================================================
// AABB Intersection
// =============================================================================
Callable<Bool(Float3, Float3, Var<Ray> &, Float, Float &, Var<HitRec> &, Var<Material> &)> HitBox =
	[](Float3 bmin, Float3 bmax, Var<Ray> &r, Float tmin, Float &closest, Var<HitRec> &rec, Var<Material> &mat) {
		Bool   hit = MakeBool(false);
		Float3 n;
		Float  tmax = closest;
		Float  tc	= tmax;

		// X planes
		Float  t	= (bmin.x() - r.origin().x()) / r.dir().x();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
				tc	= t;
				n	= Vec3(-1.0f, 0.0f, 0.0f);
				hit = true;
			});
		});
		t = (bmax.x() - r.origin().x()) / r.dir().x();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
				tc	= t;
				n	= Vec3(1.0f, 0.0f, 0.0f);
				hit = true;
			});
		});

		// Y planes
		t = (bmin.y() - r.origin().y()) / r.dir().y();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
				tc	= t;
				n	= Vec3(0.0f, -1.0f, 0.0f);
				hit = true;
			});
		});
		t = (bmax.y() - r.origin().y()) / r.dir().y();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
				tc	= t;
				n	= Vec3(0.0f, 1.0f, 0.0f);
				hit = true;
			});
		});

		// Z planes
		t = (bmin.z() - r.origin().z()) / r.dir().z();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
				tc	= t;
				n	= Vec3(0.0f, 0.0f, -1.0f);
				hit = true;
			});
		});
		t = (bmax.z() - r.origin().z()) / r.dir().z();
		If(t > tmin && t < tc, [&]() {
			Float3 p = RayAt(r, t);
			If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
				tc	= t;
				n	= Vec3(0.0f, 0.0f, 1.0f);
				hit = true;
			});
		});

		If(hit, [&]() {
			rec.t()		 = tc;
			rec.p()		 = RayAt(r, tc);
			rec.normal() = n;
			rec.mat()	 = mat;
			closest		 = tc;
		});

		Return(hit);
	};

// =============================================================================
// Scene - Cornell Box
// =============================================================================
Callable<Bool(Var<Ray> &, Float, Float, Var<HitRec> &, Int &)> HitWorld = [](Var<Ray> &r, Float tmin, Float tmax,
																																			 Var<HitRec> &rec, Int &rng) {
	Bool		  hit		= MakeBool(false);
	Float		  closest	= tmax;
	Var<HitRec>   temp;

	// Floor (white, diffuse)
	Var<Material> whiteDiff;
	whiteDiff.albedo() = MakeFloat3(0.73f, 0.73f, 0.73f);
	whiteDiff.type()	= 0;
	If(HitBox(MakeFloat3(-1.0f, -1.0f, -1.0f), MakeFloat3(1.0f, -0.75f, 1.0f), r, tmin, closest, temp, whiteDiff),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Ceiling (white, diffuse)
	If(HitBox(MakeFloat3(-1.0f, 0.75f, -1.0f), MakeFloat3(1.0f, 1.0f, 1.0f), r, tmin, closest, temp, whiteDiff), [&]() {
		hit = true;
		rec = temp;
	});

	// Back (white, diffuse)
	If(HitBox(MakeFloat3(-1.0f, -0.75f, -1.0f), MakeFloat3(1.0f, 0.75f, -0.75f), r, tmin, closest, temp, whiteDiff),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Left (red, diffuse)
	Var<Material> redDiff;
	redDiff.albedo() = MakeFloat3(0.65f, 0.05f, 0.05f);
	redDiff.type()	 = 0;
	If(HitBox(MakeFloat3(-1.0f, -0.75f, -0.75f), MakeFloat3(-0.75f, 0.75f, 1.0f), r, tmin, closest, temp, redDiff),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Right (green, diffuse)
	Var<Material> greenDiff;
	greenDiff.albedo() = MakeFloat3(0.12f, 0.45f, 0.15f);
	greenDiff.type()	= 0;
	If(HitBox(MakeFloat3(0.75f, -0.75f, -0.75f), MakeFloat3(1.0f, 0.75f, 1.0f), r, tmin, closest, temp, greenDiff),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Light (emissive)
	Var<Material> lightMat;
	lightMat.albedo() = MakeFloat3(15.0f, 15.0f, 15.0f);
	lightMat.type()	  = 2;
	If(HitBox(MakeFloat3(-0.25f, 0.74f, -0.25f), MakeFloat3(0.25f, 0.75f, 0.25f), r, tmin, closest, temp, lightMat),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Tall box (metal)
	Var<Material> metalMat;
	metalMat.albedo() = MakeFloat3(0.8f, 0.85f, 0.88f);
	metalMat.type()	  = 1;
	If(HitBox(MakeFloat3(0.15f, -0.75f, -0.4f), MakeFloat3(0.45f, -0.15f, -0.1f), r, tmin, closest, temp, metalMat),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	// Short box (diffuse)
	If(HitBox(MakeFloat3(-0.4f, -0.75f, 0.0f), MakeFloat3(-0.1f, -0.4f, 0.3f), r, tmin, closest, temp, whiteDiff),
	   [&]() {
		   hit = true;
		   rec = temp;
	   });

	Return(hit);
};

// =============================================================================
// Material Scatter
// =============================================================================
Callable<Float3(Var<HitRec> &, Var<Ray> &, Bool &, Int &)> Scatter = [](Var<HitRec> &rec, Var<Ray> &rIn,
																																							Bool &scattered, Int &rng) {
	Int matType = rec.mat().type();

	If(matType == 2, [&]() {
		scattered = false;
		Return(rec.mat().albedo());
	});

	If(matType == 1, [&]() {
		Float3 refl = Reflect(Normalize(rIn.dir()), rec.normal());
		scattered	= true;

		Return(refl + 0.2f * RandomInUnitSphere(rng));
	});

	// Diffuse
	Float3 target = rec.normal() + RandomUnitVector(rng);
	scattered	  = true;
	Return(target);
};

// =============================================================================
// Path Tracing with Next Event Estimation (NEE)
// =============================================================================
Callable<Float3(Var<Ray> &, Int &)> Trace = [](Var<Ray> &r, Int &rng) {
	Float3	 throughput = MakeFloat3(1.0f);
	Float3	 radiance	= MakeFloat3(0.0f);
	Var<Ray> cur;
	SetRay(cur, r.origin(), r.dir());

	For(0, MAX_DEPTH, [&](Int &) {
		Var<HitRec> rec;
		If(!HitWorld(cur, 0.001f, 1000.0f, rec, rng), [&]() { Break(); }).Else([&]() {
			If(rec.mat().type() == 2, [&]() {
				// Hit emitter - accumulate emission
				radiance += throughput * rec.mat().albedo();
				Break();
			}).Else([&]() {
				// Next Event Estimation for diffuse surfaces
				If(rec.mat().type() == 0, [&]() {
					// Sample a point on the light source
					Float3 lightPos = MakeFloat3(
						(Random(rng) - 0.5f) * 0.5f,
						0.745f,
						(Random(rng) - 0.5f) * 0.5f
					);

					Float3 toLight   = lightPos - rec.p();
					Float  lightDist = Length(toLight);
					Float3 lightDir  = toLight / lightDist;

					// Shadow ray
					Float3 shadowOrigin = rec.p() + rec.normal() * 0.001f;
					Var<Ray> shadowRay;
					SetRay(shadowRay, shadowOrigin, lightDir);
					Var<HitRec> srec;
					Bool		sHit = HitWorld(shadowRay, 0.001f, lightDist, srec, rng);

					If(!sHit || srec.mat().type() == 2, [&]() {
						Float  NdotL  = Max(Dot(rec.normal(), lightDir), 0.0f);
						Float3 direct = rec.mat().albedo() * MakeFloat3(15.0f) * NdotL * 0.08f;
						radiance += throughput * direct;
					});
				});

				// Indirect bounce
				Bool   scat = MakeBool(false);
				Float3 dir	= Scatter(rec, cur, scat, rng);
				If(scat, [&]() {
					throughput *= rec.mat().albedo();
					Float3 newDir = Normalize(dir);
					SetRay(cur, rec.p(), newDir);
				}).Else([&]() { Break(); });
			});
		});
	});

	Return(radiance);
};

// =============================================================================
// Main
// =============================================================================
int main() {
	try {
	using namespace GPU::Window;

	std::cout << "Cornell Box Real-time GI\n";
	std::cout << WIDTH << "x" << HEIGHT << " @ progressive rendering (1spp/frame)\n";
	std::cout << "Controls: WASD=move, QE=up/down, MouseDrag=look, ESC=exit\n\n";

	Buffer<int>		 rngState(WIDTH * HEIGHT, BufferMode::ReadWrite);
	Buffer<Vec4>	 accumBuffer(WIDTH * HEIGHT, BufferMode::ReadWrite);

	// Initialize RNG seeds
	std::vector<int> seeds(WIDTH * HEIGHT);
	for (int i = 0; i < WIDTH * HEIGHT; ++i)
		seeds[i] = i + 1;
	rngState.Upload(seeds);

	// Initialize accumulation buffer to zero
	std::vector<Vec4> zeros(WIDTH * HEIGHT, Vec4(0.0f));
	accumBuffer.Upload(zeros);

	// Display texture and window
	Texture2D<PixelFormat::RGBA8> displayTex(WIDTH, HEIGHT);
	AppWindow					  window({.width = WIDTH, .height = HEIGHT, .title = "Cornell Box Real-time GI", .vsync = true});
	TexturePresenter			  presenter(window);

	Uniform<int> frameCount(0);

	// Camera state
	Vec3 cameraPos(0.0f, 0.0f, 2.5f);
	float yaw			   = -1.570796f; // -90 deg, facing -Z
	float pitch			   = 0.0f;
	float moveSpeed		   = 0.005f;
	float mouseSensitivity = 0.003f;
	bool  mouseDragging	   = false;
	int   lastMouseX	   = -1;
	int   lastMouseY	   = -1;

	// Camera basis vectors
	Vec3 front(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
	Vec3 right = front.Cross(Vec3(0.0f, 1.0f, 0.0f)).Normalized();
	Vec3 up	   = right.Cross(front).Normalized();

	// Uniforms for GPU camera
	Uniform<Vec3> uCameraPos(cameraPos);
	Uniform<Vec3> uCameraForward(front);
	Uniform<Vec3> uCameraRight(right);
	Uniform<Vec3> uCameraUp(up);

	std::cout << "Rendering... Close window to exit.\n" << std::flush;

	// Create kernel once (avoids per-frame shader recompilation)
	Kernel::Kernel2D kernel([&](Int &px, Int &py) {
		auto img   = displayTex.Bind();
		auto state = rngState.Bind();
		auto accum = accumBuffer.Bind();

		Int idx = py * WIDTH + px;
		Int rng = state[idx];

		// Load camera uniforms (Unref to create independent local copies)
		Float3 camPos	  = Unref(uCameraPos.Load());
		Float3 camForward = Unref(uCameraForward.Load());
		Float3 camRight	  = Unref(uCameraRight.Load());
		Float3 camUp	  = Unref(uCameraUp.Load());

		// Generate camera ray with sub-pixel jitter
		Float aspect   = MakeFloat(WIDTH) / HEIGHT;
		Float fovScale = Tan(MakeFloat(20.0f * 3.14159f / 180.0f));

		Float rx = (Random(rng) - 0.5f) / WIDTH;
		Float ry = (Random(rng) - 0.5f) / HEIGHT;

		Float ndcX = (ToFloat(px) + 0.5f + rx) / WIDTH * 2.0f - 1.0f;
		Float ndcY = (ToFloat(py) + 0.5f + ry) / HEIGHT * 2.0f - 1.0f;

		Float3 rd = Normalize(camForward + camRight * ndcX * aspect * fovScale + camUp * ndcY * fovScale);
		Float3 ro = camPos;

		Var<Ray> ray;
		ray.origin() = ro;
		ray.dir() = rd;

		// Path trace 1 sample
		Float3 sampleColor = Trace(ray, rng);

		// Temporal accumulation
		Var<Vec4> prev4 = accum[idx];
		Float3	  total = MakeFloat3(
			prev4.x() + sampleColor.x(),
			prev4.y() + sampleColor.y(),
			prev4.z() + sampleColor.z()
		);
		accum[idx] = MakeFloat4(total, 1.0f);

		// Tone mapping (Reinhard + gamma)
		Int   frame	   = frameCount.Load() + 1;
		Float invFrame = 1.0f / ToFloat(frame);
		Float3 avg	   = total * invFrame;

		Float3 mapped = avg / (avg + MakeFloat3(1.0f));
		mapped		  = Pow(Clamp(mapped, 0.0f, 1.0f), MakeFloat3(1.0f / 2.2f));

		img.Write(px, HEIGHT - py, MakeFloat4(mapped.zyx(), 1.0f));

		state[idx] = rng;
	});

	while (window.IsOpen()) {
		window.PollEvents();

		bool cameraMoved = false;

		// Process window events
		WindowEvent event;
		while (window.PollEvent(event)) {
			if (std::holds_alternative<KeyEvent>(event)) {
				auto &key = std::get<KeyEvent>(event);
				if (key.key == Key::Escape && key.pressed) {
					window.Close();
				}
			} else if (std::holds_alternative<MouseButtonEvent>(event)) {
				auto &mb = std::get<MouseButtonEvent>(event);
				if (mb.button == MouseButton::Left) {
					mouseDragging = mb.pressed;
					if (!mouseDragging) {
						lastMouseX = -1;
						lastMouseY = -1;
					}
				}
			}
		}

		// Mouse look
		auto [mouseX, mouseY] = window.MousePosition();
		if (mouseDragging) {
			if (lastMouseX >= 0) {
				int dx = mouseX - lastMouseX;
				int dy = mouseY - lastMouseY;
				yaw += dx * mouseSensitivity;
				pitch -= dy * mouseSensitivity;
				pitch = std::clamp(pitch, -1.55f, 1.55f);
				cameraMoved = true;
			}
			lastMouseX = mouseX;
			lastMouseY = mouseY;
		}

		// Keyboard movement
		front = Vec3(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
		right = front.Cross(Vec3(0.0f, 1.0f, 0.0f)).Normalized();
		up	  = right.Cross(front).Normalized();

		if (window.IsKeyDown(Key::W)) {
			cameraPos = cameraPos + front * moveSpeed;
			cameraMoved = true;
		}
		if (window.IsKeyDown(Key::S)) {
			cameraPos = cameraPos - front * moveSpeed;
			cameraMoved = true;
		}
		if (window.IsKeyDown(Key::A)) {
			cameraPos = cameraPos - right * moveSpeed;
			cameraMoved = true;
		}
		if (window.IsKeyDown(Key::D)) {
			cameraPos = cameraPos + right * moveSpeed;
			cameraMoved = true;
		}
		if (window.IsKeyDown(Key::Q)) {
			cameraPos = cameraPos - up * moveSpeed;
			cameraMoved = true;
		}
		if (window.IsKeyDown(Key::E)) {
			cameraPos = cameraPos + up * moveSpeed;
			cameraMoved = true;
		}

		// Reset accumulation when camera moves
		if (cameraMoved) {
			uCameraPos		 = cameraPos;
			uCameraForward	 = front;
			uCameraRight	 = right;
			uCameraUp		 = up;
			accumBuffer.Upload(zeros);
			frameCount		 = 0;
		}

		kernel.Dispatch((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
		presenter.Present(displayTex);

		frameCount = frameCount.GetValue() + 1;
	}

	std::cout << "Rendered " << frameCount.GetValue() << " frames.\n";
	return 0;
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	} catch (...) {
		std::cerr << "Unknown error\n";
		return 1;
	}
}
