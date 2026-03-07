/**
 * @file Context.h
 * @brief Non-intrusive OpenGL context management with automatic initialization
 *
 * This singleton automatically initializes OpenGL and GLAD on first use,
 * creating a hidden window for off-screen compute shader execution.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

// Platform detection first
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// GLAD must be included before windows.h on Windows to avoid GL header conflicts
#include <GLAD/glad.h>
#include <windows.h>
using NativeGLContext = HGLRC;
#elif defined(__linux__)
// On Linux, include GLAD first, then X11
#include <GLAD/glad.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

// GLX type declarations (X11 types are now available from Xlib.h)
typedef XID						 GLXDrawable;
typedef XID						 GLXContextID;
typedef struct __GLXcontextRec	*GLXContext;
typedef struct __GLXFBConfigRec *GLXFBConfig;

// GLX_ARB_create_context constants
// GLX 1.0+ constants
#define GLX_USE_GL					1
#define GLX_BUFFER_SIZE				2
#define GLX_LEVEL					3
#define GLX_RGBA					4
#define GLX_DOUBLEBUFFER			5
#define GLX_STEREO					6
#define GLX_AUX_BUFFERS				7
#define GLX_RED_SIZE				8
#define GLX_GREEN_SIZE				9
#define GLX_BLUE_SIZE				10
#define GLX_ALPHA_SIZE				11
#define GLX_DEPTH_SIZE				12
#define GLX_STENCIL_SIZE			13
#define GLX_ACCUM_RED_SIZE			14
#define GLX_ACCUM_GREEN_SIZE		15
#define GLX_ACCUM_BLUE_SIZE			16
#define GLX_ACCUM_ALPHA_SIZE		17

// GLX 1.1+ constants
#define GLX_X_VISUAL_TYPE			0x22
#define GLX_CONFIG_CAVEAT			0x20
#define GLX_TRANSPARENT_TYPE		0x23
#define GLX_TRANSPARENT_INDEX_VALUE 0x24
#define GLX_TRANSPARENT_RED_VALUE	0x25
#define GLX_TRANSPARENT_GREEN_VALUE 0x26
#define GLX_TRANSPARENT_BLUE_VALUE	0x27
#define GLX_TRANSPARENT_ALPHA_VALUE 0x28
#define GLX_WINDOW_BIT				0x00000001
#define GLX_PIXMAP_BIT				0x00000002
#define GLX_RGBA_BIT				0x00000001
#define GLX_COLOR_INDEX_BIT			0x00000002
#define GLX_PBUFFER_BIT				0x00000004
#define GLX_TRUE_COLOR				0x8002
#define GLX_DIRECT_COLOR			0x8003
#define GLX_PSEUDO_COLOR			0x8004
#define GLX_STATIC_COLOR			0x8005
#define GLX_GRAY_SCALE				0x8006
#define GLX_STATIC_GRAY				0x8007
#define GLX_TRANSPARENT_RGB			0x8008
#define GLX_TRANSPARENT_INDEX		0x8009
#define GLX_NONE					0x8000
#define GLX_SLOW_CONFIG				0x8001
#define GLX_NON_CONFORMANT_CONFIG	0x800D

// GLX 1.3+ constants
#define GLX_X_RENDERABLE			0x8012
#define GLX_FBCONFIG_ID				0x8013
#define GLX_MAX_PBUFFER_WIDTH		0x8016
#define GLX_MAX_PBUFFER_HEIGHT		0x8017
#define GLX_MAX_PBUFFER_PIXELS		0x8018
#define GLX_VISUAL_ID				0x800B
#define GLX_SCREEN					0x800C
#define GLX_DRAWABLE_TYPE			0x8010
#define GLX_RENDER_TYPE				0x8011

// GLX_ARB_create_context constants
#ifndef GLX_CONTEXT_MAJOR_VERSION_ARB
#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef GLX_CONTEXT_MINOR_VERSION_ARB
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef GLX_CONTEXT_PROFILE_MASK_ARB
#define GLX_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef GLX_CONTEXT_CORE_PROFILE_BIT_ARB
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif

using NativeGLContext = GLXContext;

// GLX function declarations (GLAD provides these at runtime, but we need declarations for compilation)
extern "C" {
// GLAD provides glXGetProcAddressARB
typedef void (*__GLXextFuncPtr)(void);
extern __GLXextFuncPtr glXGetProcAddressARB(const GLubyte *procName);

// GLX 1.0
extern XVisualInfo	  *glXChooseVisual(Display *dpy, int screen, int *attribList);
extern GLXContext	   glXCreateContext(Display *dpy, XVisualInfo *vis, GLXContext shareList, Bool direct);
extern void			   glXDestroyContext(Display *dpy, GLXContext ctx);
extern Bool			   glXMakeCurrent(Display *dpy, GLXDrawable drawable, GLXContext ctx);
extern void			   glXSwapBuffers(Display *dpy, GLXDrawable drawable);
extern Bool			   glXQueryVersion(Display *dpy, int *maj, int *min);

// GLX 1.3+
extern GLXFBConfig	  *glXChooseFBConfig(Display *dpy, int screen, const int *attribList, int *nitems);
extern XVisualInfo	  *glXGetVisualFromFBConfig(Display *dpy, GLXFBConfig config);
// Note: XFree (not glXFree) is used to free memory from glXChooseFBConfig
}
#else
#error "Unsupported platform"
#endif

namespace GPU::Runtime {

/**
 * Singleton OpenGL context manager with automatic lazy initialization
 * Automatically creates a hidden window and OpenGL context on first use.
 * Users don't need to manually initialize anything - just use Kernel or Buffer.
 */
class Context {
public:
	~Context();

	// Non-copyable, non-movable singleton
	Context(const Context &)						= delete;

	Context &operator=(const Context &)				= delete;

	Context(Context &&)								= delete;

	Context					 &operator=(Context &&) = delete;

	/**
	 * Get the singleton instance, auto-initializing if needed
	 */
	static Context			 &GetInstance();

	/**
	 * Explicitly initialize the context (optional, called automatically)
	 * @throw std::runtime_error if initialization fails
	 */
	void					  Initialize();

	/**
	 * Check if context is already initialized
	 */
	[[nodiscard]] bool		  IsInitialized() const;

	/**
	 * Make the OpenGL context current on this thread
	 */
	void					  MakeCurrent();

	/**
	 * Release the context from current thread
	 */
	void					  MakeNoneCurrent();

	/**
	 * Get OpenGL version string
	 */
	[[nodiscard]] std::string GetVersionString() const;

	/**
	 * Check if compute shaders are supported
	 */
	[[nodiscard]] bool		  HasComputeShadersSupport() const;

	/**
	 * Get compute shader max work group size
	 */
	void					  GetMaxWorkGroupSize(int &x, int &y, int &z) const;

private:
	Context() = default;

	void InitializePlatform();

	void CleanupPlatform();

	void CreateHiddenWindow();

	void DestroyHiddenWindow();

	void SetupGLContext();

	void LoadGLAD();

public:
	/**
	 * Get the native OpenGL context handle
	 * @return The GL context handle (HGLRC on Windows, GLXContext on Linux)
	 */
#ifdef _WIN32
	[[nodiscard]] HGLRC GetGLContext() const {
		return _hglrc;
	}
#elif defined(__linux__)
	[[nodiscard]] GLXContext GetGLContext() const {
		return _glxContext;
	}

	/**
	 * Get the X11 Display connection
	 * @return The Display pointer
	 */
	[[nodiscard]] Display *GetX11Display() const {
		return _display;
	}
#endif

private:
	bool _initialized = false;

#ifdef _WIN32
	HINSTANCE _hInstance = nullptr;
	HWND	  _hwnd		 = nullptr;
	HDC		  _hdc		 = nullptr;
	HGLRC	  _hglrc	 = nullptr;
#elif defined(__linux__)
	// Linux/X11 specific members
	Display	  *_display	   = nullptr;
	Window	   _window	   = 0;
	GLXContext _glxContext = nullptr;
#endif

	// Reference count for automatic cleanup consideration
	static Context *_instance;
	static bool		_destroyed;
};

/**
 * RAII guard for making context current on a scope
 */
class ContextGuard {
public:
	explicit ContextGuard(Context &ctx) : _ctx(ctx) {
		_ctx.MakeCurrent();
	}

	~ContextGuard() {
		_ctx.MakeNoneCurrent();
	}

	ContextGuard(const ContextGuard &)			  = delete;

	ContextGuard &operator=(const ContextGuard &) = delete;

private:
	Context &_ctx;
};

/**
 * Auto-initialization helper - call this in any GPU operation entry point
 */
inline void AutoInitContext() {
	Context::GetInstance().Initialize();
}
} // namespace GPU::Runtime