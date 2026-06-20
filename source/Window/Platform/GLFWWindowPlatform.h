#pragma once

/**
 * @file GLFWWindowPlatform.h
 * @brief Internal platform abstraction using GLFW.
 */

#ifndef EASYGPU_GLFW_WINDOW_PLATFORM_H
#define EASYGPU_GLFW_WINDOW_PLATFORM_H

#include "IWindowPlatform.h"

#include <array>
#include <memory>
#include <queue>

#ifdef EASYGPU_BACKEND_VULKAN
#define GLFW_INCLUDE_VULKAN
#else
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#ifdef EASYGPU_BACKEND_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace GPU::Window {

class Swapchain;

class GLFWWindowPlatform : public IWindowPlatform {
public:
	GLFWWindowPlatform(const WindowConfig &config, std::queue<WindowEvent> &eventQueue);
	~GLFWWindowPlatform() override;

	GLFWWindowPlatform(const GLFWWindowPlatform &)							   = delete;
	GLFWWindowPlatform &operator=(const GLFWWindowPlatform &)				   = delete;
	GLFWWindowPlatform(GLFWWindowPlatform &&)								   = delete;
	GLFWWindowPlatform						 &operator=(GLFWWindowPlatform &&) = delete;

	[[nodiscard]] bool						  IsOpen() const override;
	void									  Close() override;

	[[nodiscard]] uint32_t					  Width() const override;
	[[nodiscard]] uint32_t					  Height() const override;
	void									  SetTitle(const std::string &title) override;

	void									  Present(const uint32_t *pixels, uint32_t width, uint32_t height) override;
	void									  SetOpenGLOverlay(std::function<void()> callback) override;
	void									  WaitSync() override;

	void									  PollEvents() override;
	void									  WaitEvents() override;
	bool									  PollEvent(WindowEvent &event) override;

	[[nodiscard]] bool						  IsKeyDown(int keyCode) const override;
	[[nodiscard]] bool						  IsMouseDown(int button) const override;
	[[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const override;
	[[nodiscard]] std::pair<float, float>	  MouseScroll() const override;

	[[nodiscard]] GLFWwindow				 *NativeWindow() const {
		return _window;
	}

#ifdef EASYGPU_BACKEND_VULKAN
	VkSurfaceKHR CreateVulkanSurface(VkInstance instance) const;
	Swapchain	*GetSwapchain();
#endif

private:
	void				 HandleResize(int width, int height);
	void				 HandleClose();
	void				 HandleFocus(bool focused);
	void				 HandleKeyboard(int key, int mods, bool pressed);
	void				 HandleCharInput(unsigned int codepoint);
	void				 HandleMouseButton(int button, int mods, bool pressed);
	void				 HandleMouseMove(double x, double y);
	void				 HandleMouseScroll(double dx, double dy);

	static ModifierFlags ConvertMods(int glfwMods);
	static MouseButton	 ConvertMouseButton(int glfwButton);

private:
	GLFWwindow				*_window = nullptr;
	std::queue<WindowEvent> &_eventQueue;

	uint32_t				 _width	  = 0;
	uint32_t				 _height  = 0;
	int32_t					 _mouseX  = 0;
	int32_t					 _mouseY  = 0;
	float					 _scrollX = 0.0f;
	float					 _scrollY = 0.0f;
	bool					 _isOpen  = false;
	bool					 _vsync	  = true;

	struct PixelPresenter;
	std::unique_ptr<PixelPresenter> _presenter;
	std::function<void()>			  _openGLOverlay;
};

} // namespace GPU::Window

#endif // EASYGPU_GLFW_WINDOW_PLATFORM_H
