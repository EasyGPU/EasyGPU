/**
 * @file main.cpp
 * @brief Sponza Atrium — Multi-Texture 3D via EasyGPU DSL + Texture Atlas.
 *
 * Loads Sponza OBJ + MTL, packs all diffuse textures into a single atlas,
 * remaps UVs per material, and renders with BindSampler() in the FS.
 *
 * Usage: ./sponza_renderer [path/to/Sponza/]
 * Controls: WASD = move, Mouse = look, ESC = exit
 */

#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
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
constexpr float	   FOV			 = 60.0f;
constexpr float	   NEAR_PLANE	 = 0.1f;
constexpr float	   FAR_PLANE	 = 100.0f;
constexpr float	   MODEL_SCALE	 = 0.01f;
constexpr float	   MOUSE_SENS	 = 0.003f;
constexpr float	   MOVE_SPEED	 = 0.5f;

constexpr int	   ATLAS_SIZE	 = 4096;
constexpr int	   ATLAS_GRID	 = 8; // 8x8 = 64 slots
constexpr int	   ATLAS_GUTTER	 = 2; // pixels between slots (prevents bleeding)
constexpr int	   MAX_TEX_SLOTS = ATLAS_GRID * ATLAS_GRID;

// =============================================================================
// GPU Types
// =============================================================================
EASYGPU_STRUCT(GpuVertex, (Vec3, pos), (Vec3, normal), (Vec2, uv), (Vec4, atlasTransform));
EASYGPU_STRUCT(SceneUBO, (Mat4, mvp));

// =============================================================================
// MTL Material
// =============================================================================
struct Material {
	std::string name;
	std::string texPath; // resolved texture file path
	Vec3		diffuseColor;
	int			atlasSlot = -1; // assigned atlas grid position
};

class MtlLibrary {
public:
	bool Load(const std::string &path, const std::string &texDir) {
		std::ifstream file(path);
		if (!file)
			return false;
		Material	current;
		std::string line;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#')
				continue;
			std::istringstream ss(line);
			std::string		   token;
			ss >> token;
			if (token == "newmtl") {
				if (!current.name.empty())
					_materials[current.name] = current;
				current = Material{};
				ss >> current.name;
			} else if (token == "Kd") {
				ss >> current.diffuseColor.x >> current.diffuseColor.y >> current.diffuseColor.z;
			} else if (token == "map_Kd") {
				std::string p;
				ss >> p;
				size_t s		= p.find_last_of("/\\");
				current.texPath = texDir + "/" + (s != std::string::npos ? p.substr(s + 1) : p);
			}
		}
		if (!current.name.empty())
			_materials[current.name] = current;
		return true;
	}

	const Material *Find(const std::string &name) const {
		auto it = _materials.find(name);
		return it != _materials.end() ? &it->second : nullptr;
	}

	void AssignAtlasSlots() {
		// Slot 0 = white fallback for untextured materials
		// Collect and sort by name for deterministic slot assignment
		std::vector<Material *> sorted;
		for (auto &[name, mat] : _materials)
			sorted.push_back(&mat);
		std::sort(sorted.begin(), sorted.end(), [](Material *a, Material *b) { return a->name < b->name; });
		int slot = 1;
		for (auto *mat : sorted) {
			if (!mat->texPath.empty() && slot < MAX_TEX_SLOTS)
				mat->atlasSlot = slot++;
			else if (mat->texPath.empty())
				mat->atlasSlot = 0;
		}
		std::cout << "  Materials: " << _materials.size() << "  Textured: " << (slot - 1) << "\n";
	}

	std::unordered_map<std::string, Material> &All() {
		return _materials;
	}

private:
	std::unordered_map<std::string, Material> _materials;
};

// =============================================================================
// Texture Atlas Builder
// =============================================================================
class TextureAtlas {
public:
	~TextureAtlas() {
		if (_data)
			delete[] _data;
	}

	bool Build(MtlLibrary &mtlLib) {
		mtlLib.AssignAtlasSlots();
		_data = new uint8_t[ATLAS_SIZE * ATLAS_SIZE * 4];
		// Fill entire atlas with neutral gray (friendly fallback)
		for (int i = 0; i < ATLAS_SIZE * ATLAS_SIZE * 4; i += 4) {
			_data[i]	 = 128;
			_data[i + 1] = 128;
			_data[i + 2] = 128;
			_data[i + 3] = 255;
		}
		int loaded	 = 0;

		// Fill slot 0 with white (fallback for untextured materials)
		int slotSize = ATLAS_SIZE / ATLAS_GRID;
		for (int y = 0; y < slotSize; ++y)
			for (int x = 0; x < slotSize; ++x)
				for (int c = 0; c < 4; ++c)
					_data[(y * ATLAS_SIZE + x) * 4 + c] = 255;

		std::vector<Material *> sorted;
		for (auto &[name, mat] : mtlLib.All())
			sorted.push_back(&mat);
		std::sort(sorted.begin(), sorted.end(), [](Material *a, Material *b) { return a->name < b->name; });
		for (auto *mat : sorted) {
			if (mat->atlasSlot < 0)
				continue;
			if (mat->texPath.empty())
				continue; // skip untextured (already white)

			int tw, th, ch;
			stbi_set_flip_vertically_on_load(true);
			uint8_t *pixels = stbi_load(mat->texPath.c_str(), &tw, &th, &ch, 4);
			if (!pixels) {
				mat->atlasSlot = 0;
				continue;
			}

			// Scale to fit atlas slot with gutter
			int slotSize = ATLAS_SIZE / ATLAS_GRID;
			int inner	 = slotSize - ATLAS_GUTTER * 2; // usable area inside gutter

			// Fill the full slot. Gutter texels repeat the nearest texture edge so
			// linear filtering cannot blend in the atlas background.
			for (int y = 0; y < slotSize; ++y) {
				for (int x = 0; x < slotSize; ++x) {
					int sampleX = std::clamp(x - ATLAS_GUTTER, 0, inner - 1);
					int sampleY = std::clamp(y - ATLAS_GUTTER, 0, inner - 1);
					// Bilinear sampling from source texture
					float fx = float(sampleX) * float(tw) / float(inner);
					float fy = float(sampleY) * float(th) / float(inner);
					int	  ix = int(fx), iy = int(fy);
					float u = fx - float(ix), v = fy - float(iy);
					int	  x1  = std::min(ix + 1, tw - 1);
					int	  y1  = std::min(iy + 1, th - 1);
					int	  dst = (((mat->atlasSlot / ATLAS_GRID) * slotSize + y) * ATLAS_SIZE +
							   (mat->atlasSlot % ATLAS_GRID) * slotSize + x) *
							  4;
					for (int c = 0; c < 4; ++c) {
						float v00 = pixels[(iy * tw + ix) * 4 + c];
						float v10 = pixels[(iy * tw + x1) * 4 + c];
						float v01 = pixels[(y1 * tw + ix) * 4 + c];
						float v11 = pixels[(y1 * tw + x1) * 4 + c];
						_data[dst + c] =
							uint8_t((v00 * (1 - u) * (1 - v) + v10 * u * (1 - v) + v01 * (1 - u) * v + v11 * u * v));
					}
				}
			}
			stbi_image_free(pixels);
			loaded++;
		}
		std::cout << "  Atlas: " << ATLAS_SIZE << "x" << ATLAS_SIZE << "  Loaded: " << loaded << " textures\n";
		return loaded > 0;
	}

	// Compute UV offset/scale for a material's atlas slot
	void GetUVTransform(int slot, float &ox, float &oy, float &sx, float &sy) const {
		int	  slotSize	 = ATLAS_SIZE / ATLAS_GRID;
		int	  atlasSlotX = slot % ATLAS_GRID;
		int	  atlasSlotY = slot / ATLAS_GRID;
		// Inset by gutter to avoid sampling across slot boundaries
		float g			 = float(ATLAS_GUTTER) / ATLAS_SIZE;
		float usableSize = float(slotSize - ATLAS_GUTTER * 2) / ATLAS_SIZE;
		ox				 = float(atlasSlotX * slotSize + ATLAS_GUTTER) / ATLAS_SIZE;
		oy				 = float(atlasSlotY * slotSize + ATLAS_GUTTER) / ATLAS_SIZE;
		sx				 = usableSize;
		sy				 = usableSize;
	}

	const uint8_t *Data() const {
		return _data;
	}

private:
	uint8_t *_data = nullptr;
};

// =============================================================================
// OBJ Loader (with UVs, normals, materials)
// =============================================================================
struct FaceGroup {
	std::string			  materialName;
	std::vector<uint32_t> pIdx, uIdx, nIdx;
};

class ObjMesh {
public:
	bool Load(const std::string &objPath, const std::string &baseDir, MtlLibrary &mtlLib) {
		std::ifstream file(objPath);
		if (!file)
			return false;
		std::string line, curMtl = "default";
		FaceGroup	curGrp;
		curGrp.materialName = curMtl;

		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#')
				continue;
			std::istringstream ss(line);
			std::string		   tok;
			ss >> tok;
			if (tok == "mtllib") {
				std::string m;
				ss >> m;
				mtlLib.Load(baseDir + "/" + m, baseDir + "/textures");
			} else if (tok == "v") {
				float x, y, z;
				ss >> x >> y >> z;
				_pos.push_back(Vec3(x, y, z));
			} else if (tok == "vt") {
				float u, v;
				ss >> u >> v;
				_uv.push_back(Vec2(u, v));
			} else if (tok == "vn") {
				float x, y, z;
				ss >> x >> y >> z;
				_nrm.push_back(Vec3(x, y, z));
			} else if (tok == "usemtl") {
				if (!curGrp.pIdx.empty()) {
					_groups.push_back(std::move(curGrp));
					curGrp = FaceGroup{};
				}
				ss >> curMtl;
				curGrp.materialName = curMtl;
			} else if (tok == "f") {
				std::string a, b, c, d;
				ss >> a >> b >> c >> d;
				AddFace(a, b, c, curGrp);
				if (!d.empty())
					AddFace(a, c, d, curGrp);
			}
		}
		if (!curGrp.pIdx.empty())
			_groups.push_back(std::move(curGrp));
		std::cout << "  P:" << _pos.size() << " UV:" << _uv.size() << " N:" << _nrm.size() << " Tri:" << TotalTris()
				  << "\n";
		return !_groups.empty();
	}

	void Flatten(std::vector<GpuVertex> &verts, const TextureAtlas &atlas, const MtlLibrary &mtlLib) {
		for (auto &g : _groups) {
			bool  hasUV	 = _uv.size() > 0;
			bool  hasNrm = g.nIdx.size() == g.pIdx.size();
			float ox = 0, oy = 0, sx = 1, sy = 1;
			auto *mat  = mtlLib.Find(g.materialName);
			int	  slot = mat && mat->atlasSlot >= 0 ? mat->atlasSlot : 0;
			atlas.GetUVTransform(slot, ox, oy, sx, sy);

			for (size_t i = 0; i < g.pIdx.size(); ++i) {
				Vec3 p = _pos[g.pIdx[i]];
				Vec3 n = hasNrm ? _nrm[g.nIdx[i]] : Vec3(0, 1, 0);
				Vec2 uv(0, 0);
				if (hasUV && g.uIdx[i] < _uv.size())
					uv = _uv[g.uIdx[i]];
				verts.push_back({p, n, uv, Vec4(ox, oy, sx, sy)});
			}
		}
	}

	Vec3 Center() const {
		Vec3 mn(1e10f, 1e10f, 1e10f), mx(-1e10f, -1e10f, -1e10f);
		for (auto &p : _pos) {
			mn.x = std::min(mn.x, p.x);
			mn.y = std::min(mn.y, p.y);
			mn.z = std::min(mn.z, p.z);
			mx.x = std::max(mx.x, p.x);
			mx.y = std::max(mx.y, p.y);
			mx.z = std::max(mx.z, p.z);
		}
		return Vec3((mn.x + mx.x) * .5f, (mn.y + mx.y) * .5f, (mn.z + mx.z) * .5f);
	}
	float Radius() const {
		Vec3  c = Center();
		float r = 0;
		for (auto &p : _pos)
			r = std::max(r, (p - c).Length());
		return r;
	}

private:
	size_t TotalTris() const {
		size_t n = 0;
		for (auto &g : _groups)
			n += g.pIdx.size() / 3;
		return n;
	}
	void AddFace(const std::string &a, const std::string &b, const std::string &c, FaceGroup &g) {
		int pi, ti, ni;
		ParseV(a, pi, ti, ni);
		g.pIdx.push_back(pi);
		g.uIdx.push_back(ti);
		g.nIdx.push_back(ni);
		ParseV(b, pi, ti, ni);
		g.pIdx.push_back(pi);
		g.uIdx.push_back(ti);
		g.nIdx.push_back(ni);
		ParseV(c, pi, ti, ni);
		g.pIdx.push_back(pi);
		g.uIdx.push_back(ti);
		g.nIdx.push_back(ni);
	}
	void ParseV(const std::string &s, int &pi, int &ti, int &ni) {
		pi = ti = ni = 0;
		size_t s1	 = s.find('/');
		if (s1 == std::string::npos) {
			pi = Idx(std::stoi(s), _pos.size());
			return;
		}
		pi		  = Idx(std::stoi(s.substr(0, s1)), _pos.size());
		size_t s2 = s.find('/', s1 + 1);
		if (s1 + 1 < s2)
			ti = Idx(std::stoi(s.substr(s1 + 1, s2 - s1 - 1)), _uv.size());
		if (s2 != std::string::npos && s2 + 1 < s.size())
			ni = Idx(std::stoi(s.substr(s2 + 1)), _nrm.size());
	}
	static int Idx(int i, size_t n) {
		return i > 0 ? i - 1 : (int)n + i;
	}
	std::vector<Vec3>	   _pos, _nrm;
	std::vector<Vec2>	   _uv;
	std::vector<FaceGroup> _groups;
};

// =============================================================================
// Math
// =============================================================================
static Mat4 PerspVk(float fov, float a, float n, float f) {
	float t = std::tan(fov * .5f * 3.14159265f / 180.f);
	Mat4  m;
	m.m00 = 1.f / (a * t);
	m.m11 = -1.f / t;
	m.m22 = f / (n - f);
	m.m23 = (n * f) / (n - f);
	m.m32 = -1.f;
	m.m33 = 0.f;
	return m;
}
static Mat4 CameraView(Vec3 pos, float yaw, float pitch) {
	float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
	Vec3  fwd(sy * cp, sp, -cy * cp), rt = fwd.Cross(Vec3(0, 1, 0)).Normalized(), up = rt.Cross(fwd);
	Mat4  m;
	m.m00 = rt.x;
	m.m01 = rt.y;
	m.m02 = rt.z;
	m.m03 = -rt.Dot(pos);
	m.m10 = up.x;
	m.m11 = up.y;
	m.m12 = up.z;
	m.m13 = -up.Dot(pos);
	m.m20 = -fwd.x;
	m.m21 = -fwd.y;
	m.m22 = -fwd.z;
	m.m23 = fwd.Dot(pos);
	m.m30 = 0;
	m.m31 = 0;
	m.m32 = 0;
	m.m33 = 1;
	return m;
}

// =============================================================================
int main(int argc, char **argv) {
	try {
		std::string base = (argc > 1) ? argv[1] : "Sponza";
		std::cout << "=== EasyGPU Sponza Atlas Renderer ===\nLoading " << base << "/sponza.obj ...\n";

		// ── Load Model + Materials ────────────────────────────────────
		MtlLibrary mtlLib;
		ObjMesh	   mesh;
		if (!mesh.Load(base + "/sponza.obj", base, mtlLib)) {
			std::cerr << "Failed\n";
			return 1;
		}

		// ── Build Texture Atlas ────────────────────────────────────────
		TextureAtlas		   atlas;
		bool				   hasTex = atlas.Build(mtlLib);

		// ── Flatten Vertices (with atlas UV remap) ─────────────────────
		std::vector<GpuVertex> verts;
		mesh.Flatten(verts, atlas, mtlLib);
		uint32_t vertCount = (uint32_t)verts.size();
		std::cout << "  Vertices: " << vertCount << "\n";

		// ── Model Transform ───────────────────────────────────────────
		Vec3  ctr = mesh.Center();
		float rad = mesh.Radius(), scl = MODEL_SCALE;
		std::cout << "  Center: (" << ctr.x << "," << ctr.y << "," << ctr.z << ") R:" << rad << "\n";

		// ── GPU Resources ─────────────────────────────────────────────
		Texture2D<PixelFormat::RGBA8> rt(WINDOW_WIDTH, WINDOW_HEIGHT);
		DepthBuffer					  db(WINDOW_WIDTH, WINDOW_HEIGHT);
		Buffer<GpuVertex>			  vb(verts);
		Uniform<SceneUBO>			  ubo;
		Texture2D<PixelFormat::RGBA8> tex(ATLAS_SIZE, ATLAS_SIZE, MipmapMode::Generate);
		if (hasTex)
			tex.Upload(atlas.Data());

		Mat4  proj = PerspVk(FOV, float(WINDOW_WIDTH) / WINDOW_HEIGHT, NEAR_PLANE, FAR_PLANE);
		float yaw = 0, pitch = -0.2f;
		Vec3  camPos(0, 2.f, 5.f);

		auto  Upd = [&]() {
			Mat4 T = Mat4();
			T.m00 = T.m11 = T.m22 = scl;
			T.m03				  = -ctr.x * scl;
			T.m13				  = -ctr.y * scl;
			T.m23				  = -ctr.z * scl;
			SceneUBO d;
			d.mvp = proj * CameraView(camPos, yaw, pitch) * T;
			ubo	  = d;
		};
		Upd();

		// ── Pipeline ──────────────────────────────────────────────────
		Varying<Vec2>	 vUV;
		Varying<Vec3>	 vN;
		Varying<Vec4>	 vAtlasTransform;

		GraphicsPipeline pipeline(
			"Sponza",
			[&](Float4 &gl_Position) {
				Int	 vid	= VertexIndex();
				auto buf	= vb.Bind();
				auto u		= ubo.Load();
				auto vert	= buf[vid];
				gl_Position = u.mvp() * MakeFloat4(vert.pos(), 1.f);
				vN			= Float3(Normalize(vert.normal()));
				vUV			= Float2(MakeFloat2(vert.uv().x(), vert.uv().y()));
				vAtlasTransform =
					Float4(MakeFloat4(vert.atlasTransform().x(), vert.atlasTransform().y(),
									 vert.atlasTransform().z(), vert.atlasTransform().w()));
			},
			[&](Float4 &fragColor) {
				Float3 N	 = Float3(Normalize(Float3(vN)));
				Float2 tiled = Fract(Float2(vUV));
				Float4 atlas = vAtlasTransform;
				Float2 uv	 = MakeFloat2(atlas.x() + tiled.x() * atlas.z(), atlas.y() + tiled.y() * atlas.w());
				Float2 scale	 = MakeFloat2(atlas.z(), atlas.w());
				Float2 dx	 = Ddx(Float2(vUV)) * scale;
				Float2 dy	 = Ddy(Float2(vUV)) * scale;
				Float  lit	 = Max(Dot(N, MakeFloat3(0.3f, 0.6f, 0.4f)), 0.2f);

				auto   tx  = tex.BindSampler();
				Float4 tc  = tx.SampleGrad(uv, dx, dy);
				fragColor  = MakeFloat4(tc.x() * lit, tc.y() * lit, tc.z() * lit, 1.f);
			});

		std::cout << "Pipeline ready.  WASD=move  Mouse=look  ESC=exit\n";

		// ── Window ────────────────────────────────────────────────────
		GPU::Window::AppWindow		  window({.width	 = WINDOW_WIDTH,
											  .height	 = WINDOW_HEIGHT,
											  .title	 = "EasyGPU — Sponza Atrium (Textured)",
											  .resizable = true,
											  .vsync	 = true});
		GPU::Window::TexturePresenter presenter(window);
		float						  lmx = 0, lmy = 0;
		bool						  fm = true;

		while (window.IsOpen()) {
			window.PollEvents();
			GPU::Window::WindowEvent ev;
			while (window.PollEvent(ev))
				if (auto *k = std::get_if<GPU::Window::KeyEvent>(&ev))
					if (k->key == GPU::Window::Key::Escape && k->pressed)
						window.Close();
			auto [mx, my] = window.MousePosition();
			if (!fm) {
				yaw	  -= (mx - lmx) * MOUSE_SENS;
				pitch -= (my - lmy) * MOUSE_SENS;
				pitch  = std::max(-1.5f, std::min(1.5f, pitch));
			}
			fm		 = false;
			lmx		 = (float)mx;
			lmy		 = (float)my;
			float cy = std::cos(yaw), sy = std::sin(yaw);
			Vec3  fwd(sy, 0, -cy), right(cy, 0, sy);
			if (window.IsKeyDown(GPU::Window::Key::W))
				camPos = camPos + fwd * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::S))
				camPos = camPos - fwd * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::A))
				camPos = camPos - right * MOVE_SPEED;
			if (window.IsKeyDown(GPU::Window::Key::D))
				camPos = camPos + right * MOVE_SPEED;
			Upd();
			pipeline.Draw(rt, db, vertCount, true);
			presenter.Present(rt);
		}
		std::cout << "Exiting.\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
}
