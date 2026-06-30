/**
 * @file GLFWWindowPlatform.cpp
 * @brief Implementation of GLFW-based window platform backend.
 */

#include "GLFWWindowPlatform.h"

#ifdef EASYGPU_BACKEND_OPENGL
#include <GLAD/glad.h>
#endif

#ifdef EASYGPU_BACKEND_VULKAN
#include "../Swapchain.h"
#include <Backend/VulkanBackend.h>
#endif
#include <Runtime/Context.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::Window {

namespace {

int	 g_glfwRefCount = 0;

void EnsureGLFWInitialized() {
	if (g_glfwRefCount == 0 && !glfwInit()) {
		throw std::runtime_error("Failed to initialize GLFW");
	}
	++g_glfwRefCount;
}

void ReleaseGLFW() {
	if (g_glfwRefCount > 0) {
		--g_glfwRefCount;
		if (g_glfwRefCount == 0) {
			glfwTerminate();
		}
	}
}

void RegisterVulkanInstanceExtensionProvider() {
#ifdef EASYGPU_BACKEND_VULKAN
	GPU::Backend::VulkanBackend::RegisterInstanceExtensionProvider([]() {
		std::vector<const char *> result;
		result.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef __APPLE__
		result.push_back("VK_EXT_metal_surface");
#elif defined(_WIN32)
		result.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__linux__)
		result.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif
		return result;
	});
#endif
}

const bool g_registeredVulkanInstanceExtensionProvider = []() {
	RegisterVulkanInstanceExtensionProvider();
	return true;
}();

GLFWWindowPlatform *GetPlatform(GLFWwindow *window) {
	return static_cast<GLFWWindowPlatform *>(glfwGetWindowUserPointer(window));
}

#ifdef EASYGPU_BACKEND_VULKAN
void CheckVk(VkResult result, const char *operation) {
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string(operation) + " failed");
	}
}
#endif

} // namespace

	struct GLFWWindowPlatform::PixelPresenter {
	explicit PixelPresenter(GLFWWindowPlatform &platform) : owner(platform) {
#ifdef EASYGPU_BACKEND_VULKAN
		GPU::Runtime::AutoInitContext();
		auto &backend = GPU::Runtime::Context::GetBackend<GPU::Backend::VulkanBackend>();
		surface		  = owner.CreateVulkanSurface(backend.Instance());
		swapchain.Create(SwapchainConfig{.instance		   = backend.Instance(),
										 .physicalDevice   = backend.PhysicalDevice(),
										 .device		   = backend.Device(),
										 .surface		   = surface,
										 .queueFamilyIndex = backend.QueueFamilyIndex(),
										 .queue			   = backend.Queue(),
										 .width			   = owner.Width(),
										 .height		   = owner.Height(),
										 .vsync			   = owner._vsync});
#endif
#ifdef EASYGPU_BACKEND_OPENGL
		CreateOpenGLResources();
#endif
	}

	~PixelPresenter() {
#ifdef EASYGPU_BACKEND_VULKAN
		swapchain.Destroy();
		if (surface) {
			GPU::Runtime::AutoInitContext();
			auto &backend = GPU::Runtime::Context::GetBackend<GPU::Backend::VulkanBackend>();
			vkDestroySurfaceKHR(backend.Instance(), surface, nullptr);
		}
#endif
#ifdef EASYGPU_BACKEND_OPENGL
		DestroyOpenGLResources();
#endif
	}

	void Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
#ifdef EASYGPU_BACKEND_VULKAN
		swapchain.PresentPixels(pixels, width, height);
#endif
#ifdef EASYGPU_BACKEND_OPENGL
		PresentOpenGL(pixels, width, height);
#endif
	}

	GLFWWindowPlatform &owner;

#ifdef EASYGPU_BACKEND_VULKAN
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	Swapchain	 swapchain;
#endif

#ifdef EASYGPU_BACKEND_OPENGL
	GLuint	 texture	   = 0;
	GLuint	 vao		   = 0;
	GLuint	 vbo		   = 0;
	GLuint	 program	   = 0;
	uint32_t textureWidth  = 0;
	uint32_t textureHeight = 0;

	GLuint	 Compile(GLenum type, const char *source) {
		GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &source, nullptr);
		glCompileShader(shader);
		GLint ok = GL_FALSE;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
		if (!ok) {
			glDeleteShader(shader);
			throw std::runtime_error("Failed to compile GLFW pixel presenter shader");
		}
		return shader;
	}

	void CreateOpenGLResources() {
		glfwMakeContextCurrent(owner._window);
		if (!gladLoadGL()) {
			throw std::runtime_error("Failed to load OpenGL functions for GLFW window");
		}

		const char *vertexSource   = R"glsl(
			#version 330 core
			layout(location = 0) in vec2 inPos;
			layout(location = 1) in vec2 inUV;
			out vec2 uv;
			void main() {
				uv = inUV;
				gl_Position = vec4(inPos, 0.0, 1.0);
			}
		)glsl";
		const char *fragmentSource = R"glsl(
			#version 330 core
			in vec2 uv;
			out vec4 outColor;
			uniform sampler2D imageTexture;
			void main() {
				outColor = texture(imageTexture, uv);
			}
		)glsl";

		GLuint		vs			   = Compile(GL_VERTEX_SHADER, vertexSource);
		GLuint		fs			   = Compile(GL_FRAGMENT_SHADER, fragmentSource);
		program					   = glCreateProgram();
		glAttachShader(program, vs);
		glAttachShader(program, fs);
		glLinkProgram(program);
		glDeleteShader(vs);
		glDeleteShader(fs);
		GLint linked = GL_FALSE;
		glGetProgramiv(program, GL_LINK_STATUS, &linked);
		if (!linked) {
			glDeleteProgram(program);
			program = 0;
			throw std::runtime_error("Failed to link GLFW pixel presenter shader program");
		}
		glUseProgram(program);
		glUniform1i(glGetUniformLocation(program, "imageTexture"), 0);

		float vertices[] = {
			-1.0f, -1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
			-1.0f, 1.0f,  0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 0.0f,
		};
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));

		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void DestroyOpenGLResources() {
		if (!owner._window) {
			return;
		}
		glfwMakeContextCurrent(owner._window);
		if (texture) {
			glDeleteTextures(1, &texture);
		}
		if (vbo) {
			glDeleteBuffers(1, &vbo);
		}
		if (vao) {
			glDeleteVertexArrays(1, &vao);
		}
		if (program) {
			glDeleteProgram(program);
		}
		glfwMakeContextCurrent(nullptr);
	}

	void PresentOpenGL(const uint32_t *pixels, uint32_t width, uint32_t height) {
		if (!pixels || width == 0 || height == 0) {
			return;
		}
		glfwMakeContextCurrent(owner._window);
		if (textureWidth != width || textureHeight != height) {
			textureWidth  = width;
			textureHeight = height;
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0,
						 GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		} else {
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height), GL_RGBA,
							GL_UNSIGNED_BYTE, pixels);
		}

		int framebufferWidth  = 0;
		int framebufferHeight = 0;
		glfwGetFramebufferSize(owner._window, &framebufferWidth, &framebufferHeight);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, framebufferWidth, framebufferHeight);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(program);
		glBindVertexArray(vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		if (owner._openGLOverlay) {
			owner._openGLOverlay();
			owner._openGLOverlay = {};
		}
		glfwSwapBuffers(owner._window);
	}
#endif
};

GLFWWindowPlatform::GLFWWindowPlatform(const WindowConfig &config, std::queue<WindowEvent> &eventQueue)
	: _eventQueue(eventQueue), _width(config.width), _height(config.height), _vsync(config.vsync) {
	(void)g_registeredVulkanInstanceExtensionProvider;
	EnsureGLFWInitialized();

#ifdef EASYGPU_BACKEND_VULKAN
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif
#ifdef EASYGPU_BACKEND_OPENGL
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
#endif
	glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_VISIBLE, config.visible ? GLFW_TRUE : GLFW_FALSE);

	_window = glfwCreateWindow(static_cast<int>(config.width), static_cast<int>(config.height), config.title.c_str(),
							   nullptr, nullptr);
	if (!_window) {
		ReleaseGLFW();
		throw std::runtime_error("Failed to create GLFW window");
	}

	glfwSetWindowUserPointer(_window, this);
	glfwSetWindowSizeCallback(_window, [](GLFWwindow *window, int width, int height) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleResize(width, height);
		}
	});
	glfwSetWindowCloseCallback(_window, [](GLFWwindow *window) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleClose();
		}
	});
	glfwSetWindowFocusCallback(_window, [](GLFWwindow *window, int focused) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleFocus(focused == GLFW_TRUE);
		}
	});
	glfwSetKeyCallback(_window, [](GLFWwindow *window, int key, int, int action, int mods) {
		if (auto *platform = GetPlatform(window); platform && action != GLFW_REPEAT) {
			platform->HandleKeyboard(key, mods, action == GLFW_PRESS);
		}
	});
	glfwSetCharCallback(_window, [](GLFWwindow *window, unsigned int codepoint) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleCharInput(codepoint);
		}
	});
	glfwSetMouseButtonCallback(_window, [](GLFWwindow *window, int button, int action, int mods) {
		if (auto *platform = GetPlatform(window); platform && action != GLFW_REPEAT) {
			platform->HandleMouseButton(button, mods, action == GLFW_PRESS);
		}
	});
	glfwSetCursorPosCallback(_window, [](GLFWwindow *window, double x, double y) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleMouseMove(x, y);
		}
	});
	glfwSetScrollCallback(_window, [](GLFWwindow *window, double dx, double dy) {
		if (auto *platform = GetPlatform(window)) {
			platform->HandleMouseScroll(dx, dy);
		}
	});

	double mouseX = 0.0;
	double mouseY = 0.0;
	glfwGetCursorPos(_window, &mouseX, &mouseY);
	_mouseX	   = static_cast<int32_t>(mouseX);
	_mouseY	   = static_cast<int32_t>(mouseY);
	_isOpen	   = true;

	_presenter = std::make_unique<PixelPresenter>(*this);
}

GLFWWindowPlatform::~GLFWWindowPlatform() {
	_presenter.reset();
	if (_window) {
		glfwDestroyWindow(_window);
		_window = nullptr;
	}
	ReleaseGLFW();
}

bool GLFWWindowPlatform::IsOpen() const {
	return _isOpen && _window && !glfwWindowShouldClose(_window);
}

void GLFWWindowPlatform::Close() {
	_isOpen = false;
	if (_window) {
		glfwSetWindowShouldClose(_window, GLFW_TRUE);
	}
}

uint32_t GLFWWindowPlatform::Width() const {
	return _width;
}

uint32_t GLFWWindowPlatform::Height() const {
	return _height;
}

void GLFWWindowPlatform::SetTitle(const std::string &title) {
	if (_window) {
		glfwSetWindowTitle(_window, title.c_str());
	}
}

void GLFWWindowPlatform::Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
	if (!IsOpen() || !_presenter) {
		return;
	}
	_presenter->Present(pixels, width, height);
}

void GLFWWindowPlatform::SetOpenGLOverlay(std::function<void()> callback) {
#ifdef EASYGPU_BACKEND_OPENGL
	_openGLOverlay = std::move(callback);
#else
	(void)callback;
#endif
}

void GLFWWindowPlatform::WaitSync() {
	glfwSwapInterval(_vsync ? 1 : 0);
}

void GLFWWindowPlatform::PollEvents() {
	if (!_window || !_isOpen) {
		return;
	}
	_scrollX = 0.0f;
	_scrollY = 0.0f;
	glfwPollEvents();
	if (glfwWindowShouldClose(_window)) {
		_isOpen = false;
	}
}

void GLFWWindowPlatform::WaitEvents() {
	if (!_window || !_isOpen) {
		return;
	}
	_scrollX = 0.0f;
	_scrollY = 0.0f;
	glfwWaitEvents();
	if (glfwWindowShouldClose(_window)) {
		_isOpen = false;
	}
}

bool GLFWWindowPlatform::PollEvent(WindowEvent &event) {
	if (_eventQueue.empty()) {
		return false;
	}
	event = _eventQueue.front();
	_eventQueue.pop();
	return true;
}

bool GLFWWindowPlatform::IsKeyDown(int keyCode) const {
	return _window && keyCode >= 0 && glfwGetKey(_window, keyCode) == GLFW_PRESS;
}

bool GLFWWindowPlatform::IsMouseDown(int button) const {
	if (!_window) {
		return false;
	}
	int glfwButton = GLFW_MOUSE_BUTTON_LEFT;
	switch (static_cast<MouseButton>(button)) {
	case MouseButton::Left:
		glfwButton = GLFW_MOUSE_BUTTON_LEFT;
		break;
	case MouseButton::Right:
		glfwButton = GLFW_MOUSE_BUTTON_RIGHT;
		break;
	case MouseButton::Middle:
		glfwButton = GLFW_MOUSE_BUTTON_MIDDLE;
		break;
	default:
		glfwButton = GLFW_MOUSE_BUTTON_4 + (button - static_cast<int>(MouseButton::Button4));
		break;
	}
	return glfwGetMouseButton(_window, glfwButton) == GLFW_PRESS;
}

std::pair<int32_t, int32_t> GLFWWindowPlatform::MousePosition() const {
	return {_mouseX, _mouseY};
}

std::pair<float, float> GLFWWindowPlatform::MouseScroll() const {
	return {_scrollX, _scrollY};
}

#ifdef EASYGPU_BACKEND_VULKAN
VkSurfaceKHR GLFWWindowPlatform::CreateVulkanSurface(VkInstance instance) const {
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	CheckVk(glfwCreateWindowSurface(instance, _window, nullptr, &surface), "glfwCreateWindowSurface");
	return surface;
}

Swapchain *GLFWWindowPlatform::GetSwapchain() {
	return _presenter ? &_presenter->swapchain : nullptr;
}
#endif

void GLFWWindowPlatform::HandleResize(int width, int height) {
	_width	= static_cast<uint32_t>(std::max(width, 0));
	_height = static_cast<uint32_t>(std::max(height, 0));
#ifdef EASYGPU_BACKEND_VULKAN
	if (_presenter) {
		_presenter->swapchain.Recreate(_width, _height);
	}
#endif
	_eventQueue.push(WindowResizeEvent{.width = _width, .height = _height});
	if (resizeCallback) {
		resizeCallback(_width, _height);
	}
}

void GLFWWindowPlatform::HandleClose() {
	bool allowClose = true;
	if (closeCallback) {
		allowClose = closeCallback();
	}
	if (allowClose) {
		_isOpen = false;
		_eventQueue.push(WindowCloseEvent{});
	} else {
		glfwSetWindowShouldClose(_window, GLFW_FALSE);
	}
}

void GLFWWindowPlatform::HandleFocus(bool focused) {
	_eventQueue.push(WindowFocusEvent{.focused = focused});
	if (focusCallback) {
		focusCallback(focused);
	}
}

void GLFWWindowPlatform::HandleKeyboard(int key, int mods, bool pressed) {
	_eventQueue.push(KeyEvent{.key = static_cast<Key>(key), .pressed = pressed, .modifiers = ConvertMods(mods)});
}

void GLFWWindowPlatform::HandleCharInput(unsigned int codepoint) {
	_eventQueue.push(CharInputEvent{.codepoint = codepoint});
}

void GLFWWindowPlatform::HandleMouseButton(int button, int mods, bool pressed) {
	_eventQueue.push(MouseButtonEvent{.button	 = ConvertMouseButton(button),
									  .pressed	 = pressed,
									  .x		 = _mouseX,
									  .y		 = _mouseY,
									  .modifiers = ConvertMods(mods)});
}

void GLFWWindowPlatform::HandleMouseMove(double x, double y) {
	const int32_t newX = static_cast<int32_t>(x);
	const int32_t newY = static_cast<int32_t>(y);
	const int32_t dx   = newX - _mouseX;
	const int32_t dy   = newY - _mouseY;
	_mouseX			   = newX;
	_mouseY			   = newY;
	_eventQueue.push(MouseMoveEvent{.x = _mouseX, .y = _mouseY, .dx = dx, .dy = dy});
}

void GLFWWindowPlatform::HandleMouseScroll(double dx, double dy) {
	_scrollX = static_cast<float>(dx);
	_scrollY = static_cast<float>(dy);
	_eventQueue.push(MouseScrollEvent{.dx = _scrollX, .dy = _scrollY});
}

ModifierFlags GLFWWindowPlatform::ConvertMods(int glfwMods) {
	ModifierFlags result = ModifierFlags::None;
	if (glfwMods & GLFW_MOD_SHIFT) {
		result = result | ModifierFlags::Shift;
	}
	if (glfwMods & GLFW_MOD_CONTROL) {
		result = result | ModifierFlags::Ctrl;
	}
	if (glfwMods & GLFW_MOD_ALT) {
		result = result | ModifierFlags::Alt;
	}
	if (glfwMods & GLFW_MOD_SUPER) {
		result = result | ModifierFlags::Super;
	}
	if (glfwMods & GLFW_MOD_CAPS_LOCK) {
		result = result | ModifierFlags::CapsLock;
	}
	if (glfwMods & GLFW_MOD_NUM_LOCK) {
		result = result | ModifierFlags::NumLock;
	}
	return result;
}

MouseButton GLFWWindowPlatform::ConvertMouseButton(int glfwButton) {
	switch (glfwButton) {
	case GLFW_MOUSE_BUTTON_LEFT:
		return MouseButton::Left;
	case GLFW_MOUSE_BUTTON_RIGHT:
		return MouseButton::Right;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		return MouseButton::Middle;
	case GLFW_MOUSE_BUTTON_4:
		return MouseButton::Button4;
	case GLFW_MOUSE_BUTTON_5:
		return MouseButton::Button5;
	case GLFW_MOUSE_BUTTON_6:
		return MouseButton::Button6;
	case GLFW_MOUSE_BUTTON_7:
		return MouseButton::Button7;
	default:
		return MouseButton::Left;
	}
}

} // namespace GPU::Window
