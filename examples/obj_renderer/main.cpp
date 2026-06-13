/**
 * @file main.cpp
 * @brief OBJ Model Viewer — GPU Rasterization via EasyGPU DSL.
 *
 * Usage: ./obj_renderer [model.obj]
 * Controls: WASD = move, Mouse = look, ESC = exit
 */

#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace GPU;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Kernel;

// =============================================================================
// Configuration
// =============================================================================
constexpr uint32_t WINDOW_WIDTH	 = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr float	   FOV			 = 70.0f;
constexpr float	   NEAR_PLANE	 = 0.001f;
constexpr float	   FAR_PLANE	 = 100.0f;
constexpr float	   MODEL_SCALE	 = 2.0f; // target size after scaling
constexpr float	   MOUSE_SENS	 = 0.003f;
constexpr float	   MOVE_SPEED	 = 0.02f;

// =============================================================================
// GPU Types
// =============================================================================
EASYGPU_STRUCT(GpuVertex, (Vec3, pos), (Vec3, normal));
EASYGPU_STRUCT(SceneUBO, (Mat4, mvp));

// =============================================================================
// OBJ Loader
// =============================================================================
class ObjMesh {
public:
	bool Load(const std::string &path) {
		std::ifstream file(path);
		if (!file)
			return false;

		std::string line;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#')
				continue;

			std::istringstream ss(line);
			std::string		   token;
			ss >> token;

			if (token == "v")
				ParsePosition(ss);
			else if (token == "vn")
				ParseNormal(ss);
			else if (token == "f")
				ParseFace(ss);
		}
		return !_posIdx.empty();
	}

	void Flatten(std::vector<GpuVertex> &out) const {
		bool hasNrm = _nrmIdx.size() == _posIdx.size();
		out.reserve(_posIdx.size());
		for (size_t i = 0; i < _posIdx.size(); ++i)
			out.push_back({_positions[_posIdx[i]], hasNrm ? _normals[_nrmIdx[i]] : Vec3(0, 1, 0)});
	}

	Vec3 Center() const {
		Vec3 mn(1e10f, 1e10f, 1e10f), mx(-1e10f, -1e10f, -1e10f);
		for (auto &p : _positions) {
			mn.x = std::min(mn.x, p.x);
			mn.y = std::min(mn.y, p.y);
			mn.z = std::min(mn.z, p.z);
			mx.x = std::max(mx.x, p.x);
			mx.y = std::max(mx.y, p.y);
			mx.z = std::max(mx.z, p.z);
		}
		return Vec3((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f, (mn.z + mx.z) * 0.5f);
	}

	float Radius() const {
		Vec3  c = Center();
		float r = 0;
		for (auto &p : _positions)
			r = std::max(r, (p - c).Length());
		return r;
	}

	size_t PositionCount() const {
		return _positions.size();
	}
	size_t TriangleCount() const {
		return _posIdx.size() / 3;
	}

private:
	void ParsePosition(std::istringstream &ss) {
		float x, y, z;
		ss >> x >> y >> z;
		_positions.push_back(Vec3(x, y, z));
	}

	void ParseNormal(std::istringstream &ss) {
		float x, y, z;
		ss >> x >> y >> z;
		_normals.push_back(Vec3(x, y, z));
	}

	void ParseFace(std::istringstream &ss) {
		std::string a, b, c, d;
		ss >> a >> b >> c >> d;
		AddTriangle(a, b, c);
		if (!d.empty())
			AddTriangle(a, c, d);
	}

	void AddTriangle(const std::string &v0, const std::string &v1, const std::string &v2) {
		int p0, n0, p1, n1, p2, n2;
		ParseVertex(v0, p0, n0);
		ParseVertex(v1, p1, n1);
		ParseVertex(v2, p2, n2);
		_posIdx.push_back(p0 > 0 ? p0 - 1 : (int)_positions.size() + p0);
		_posIdx.push_back(p1 > 0 ? p1 - 1 : (int)_positions.size() + p1);
		_posIdx.push_back(p2 > 0 ? p2 - 1 : (int)_positions.size() + p2);
		if (!_normals.empty() && n0 > 0 && n1 > 0 && n2 > 0) {
			_nrmIdx.push_back(n0 - 1);
			_nrmIdx.push_back(n1 - 1);
			_nrmIdx.push_back(n2 - 1);
		}
	}

	static void ParseVertex(const std::string &s, int &posIdx, int &nrmIdx) {
		posIdx = nrmIdx = 0;
		size_t s1		= s.find('/');
		if (s1 == std::string::npos) {
			posIdx = std::stoi(s);
			return;
		}
		posIdx	  = std::stoi(s.substr(0, s1));
		size_t s2 = s.find('/', s1 + 1);
		if (s2 != std::string::npos)
			nrmIdx = std::stoi(s.substr(s2 + 1));
		else if (s1 + 1 < s.size())
			nrmIdx = std::stoi(s.substr(s1 + 1));
	}

	std::vector<Vec3>	  _positions, _normals;
	std::vector<uint32_t> _posIdx, _nrmIdx;
};

// =============================================================================
// Math — Vulkan perspective + free camera
// =============================================================================
static Mat4 PerspectiveVk(float fov, float aspect, float n, float f) {
	float t = std::tan(fov * 0.5f * 3.14159265f / 180.0f);
	Mat4  m;
	m.m00 = 1.0f / (aspect * t);
	m.m11 = -1.0f / t;
	m.m22 = f / (n - f);
	m.m23 = (n * f) / (n - f);
	m.m32 = -1.0f;
	m.m33 = 0.0f;
	return m;
}

static Mat4 CameraView(Vec3 pos, float yaw, float pitch) {
	float cy = std::cos(yaw), sy = std::sin(yaw);
	float cp = std::cos(pitch), sp = std::sin(pitch);
	Vec3  forward(sy * cp, sp, -cy * cp);
	Vec3  right = forward.Cross(Vec3(0, 1, 0)).Normalized();
	Vec3  up	= right.Cross(forward);

	Mat4  m;
	m.m00 = right.x;
	m.m01 = right.y;
	m.m02 = right.z;
	m.m03 = -right.Dot(pos);
	m.m10 = up.x;
	m.m11 = up.y;
	m.m12 = up.z;
	m.m13 = -up.Dot(pos);
	m.m20 = -forward.x;
	m.m21 = -forward.y;
	m.m22 = -forward.z;
	m.m23 = forward.Dot(pos);
	m.m30 = 0;
	m.m31 = 0;
	m.m32 = 0;
	m.m33 = 1;
	return m;
}

// =============================================================================
int main(int argc, char **argv) {
	try {
		const char *path = (argc > 1) ? argv[1] : "sponza.obj";

		// ── Load OBJ ──────────────────────────────────────────────────
		std::cout << "=== EasyGPU OBJ Viewer ===\nLoading " << path << " ...\n";

		ObjMesh mesh;
		if (!mesh.Load(path)) {
			std::cerr << "Failed to load: " << path << "\n";
			return 1;
		}
		std::cout << "  Positions: " << mesh.PositionCount() << "  Triangles: " << mesh.TriangleCount() << "\n";

		// ── Flatten vertices (non-indexed SSBO) ───────────────────────
		std::vector<GpuVertex> verts;
		mesh.Flatten(verts);
		uint32_t vertCount = (uint32_t)verts.size();
		std::cout << "  Vertices: " << vertCount << "\n";

		// ── Model transform ───────────────────────────────────────────
		Vec3  center = mesh.Center();
		float radius = mesh.Radius();
		float scale	 = MODEL_SCALE / radius;
		std::cout << "  Center: (" << center.x << ", " << center.y << ", " << center.z << ")  Radius: " << radius
				  << "  Scale: " << scale << "\n";

		// ── GPU Resources ─────────────────────────────────────────────
		Texture2D<PixelFormat::RGBA8> renderTarget(WINDOW_WIDTH, WINDOW_HEIGHT);
		DepthBuffer					  depthBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);
		Buffer<GpuVertex>			  vertexBuffer(verts);
		Uniform<SceneUBO>			  uniform;

		Mat4  projection = PerspectiveVk(FOV, float(WINDOW_WIDTH) / WINDOW_HEIGHT, NEAR_PLANE, FAR_PLANE);

		// ── Camera ────────────────────────────────────────────────────
		float yaw = 0.0f, pitch = 0.0f;
		Vec3  position(0.0f, 0.3f, 0.0f); // start at model center, slightly above floor

		auto  UpdateUniform = [&]() {
			Mat4 T = Mat4();
			T.m00 = T.m11 = T.m22 = scale;
			T.m03				  = -center.x * scale;
			T.m13				  = -center.y * scale;
			T.m23				  = -center.z * scale;
			SceneUBO data;
			data.mvp = projection * CameraView(position, yaw, pitch) * T;
			uniform	 = data;
		};
		UpdateUniform();

		// ── Pipeline ──────────────────────────────────────────────────
		Varying<Vec3>	 varyingColor;

		GraphicsPipeline pipeline(
			"OBJ",
			[&](Float4 &gl_Position) {
				Int	 vid	= VertexIndex();
				auto buf	= vertexBuffer.Bind();
				auto u		= uniform.Load();
				auto vert	= buf[vid];

				gl_Position = u.mvp() * MakeFloat4(vert.pos(), 1.0f);

				Float3 N(Normalize(vert.normal()));
				Float  diff	 = Max(Dot(N, MakeFloat3(0.4f, 0.6f, 0.7f)), 0.15f);
				varyingColor = Float3(MakeFloat3(diff * 0.9f + 0.1f, diff * 0.55f + 0.05f, diff * 0.35f + 0.08f));
			},
			[&](Float4 &fragColor) {
				Float3 c  = varyingColor;
				fragColor = MakeFloat4(c.x(), c.y(), c.z(), 1.0f);
			});

		std::cout << "Pipeline ready.\n"
				  << "  WASD = move  Mouse = look  ESC = exit\n";

		// ── Window ────────────────────────────────────────────────────
		GPU::Window::AppWindow		  window({.width	 = WINDOW_WIDTH,
											  .height	 = WINDOW_HEIGHT,
											  .title	 = "EasyGPU 3D — OBJ Model",
											  .resizable = true,
											  .vsync	 = true});
		GPU::Window::TexturePresenter presenter(window);

		float						  lastMouseX = 0, lastMouseY = 0;
		bool						  firstMouse = true;

		while (window.IsOpen()) {
			window.PollEvents();

			// Process events
			GPU::Window::WindowEvent event;
			while (window.PollEvent(event))
				if (auto *key = std::get_if<GPU::Window::KeyEvent>(&event))
					if (key->key == GPU::Window::Key::Escape && key->pressed)
						window.Close();

			// Mouse look
			auto [mx, my] = window.MousePosition();
			if (!firstMouse) {
				yaw	  -= (mx - lastMouseX) * MOUSE_SENS;
				pitch -= (my - lastMouseY) * MOUSE_SENS;
				pitch  = std::max(-1.5f, std::min(1.5f, pitch));
			}
			firstMouse = false;
			lastMouseX = (float)mx;
			lastMouseY = (float)my;

			// WASD movement in camera-local space
			float cy = std::cos(yaw), sy = std::sin(yaw);
			Vec3  forward(sy, 0, -cy), right(cy, 0, sy);
			if (window.IsKeyDown(GPU::Window::Key::W))
				position = position + forward * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::S))
				position = position - forward * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::A))
				position = position - right * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::D))
				position = position + right * MOVE_SPEED;

			UpdateUniform();
			pipeline.Draw(renderTarget, depthBuffer, vertCount, true);
			presenter.Present(renderTarget);
		}

		std::cout << "Exiting.\n";
		return 0;

	} catch (const std::exception &e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
}
