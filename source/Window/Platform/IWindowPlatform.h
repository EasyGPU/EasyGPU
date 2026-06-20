#pragma once

#ifndef EASYGPU_IWINDOW_PLATFORM_H
#define EASYGPU_IWINDOW_PLATFORM_H

#include <Window/WindowConfig.h>
#include <Window/WindowEvents.h>

#include <cstdint>
#include <functional>
#include <string>

namespace GPU::Window {

class IWindowPlatform {
public:
	virtual ~IWindowPlatform()																		= default;

	[[nodiscard]] virtual bool	   IsOpen() const													= 0;
	virtual void				   Close()															= 0;

	[[nodiscard]] virtual uint32_t Width() const													= 0;
	[[nodiscard]] virtual uint32_t Height() const													= 0;
	virtual void				   SetTitle(const std::string &title)								= 0;

	virtual void				   Present(const uint32_t *pixels, uint32_t width, uint32_t height) = 0;
	virtual void				   SetOpenGLOverlay(std::function<void()> callback) {
		(void)callback;
	}
	virtual void				   WaitSync()														= 0;

	virtual void				   PollEvents()														= 0;
	virtual void				   WaitEvents()														= 0;
	virtual bool				   PollEvent(WindowEvent &event)									= 0;

	[[nodiscard]] virtual bool	   IsKeyDown(int keyCode) const										= 0;
	[[nodiscard]] virtual bool	   IsMouseDown(int button) const									= 0;
	[[nodiscard]] virtual std::pair<int32_t, int32_t> MousePosition() const							= 0;
	[[nodiscard]] virtual std::pair<float, float>	  MouseScroll() const							= 0;

	std::function<void(uint32_t, uint32_t)>			  resizeCallback;
	std::function<bool()>							  closeCallback;
	std::function<void(bool)>						  focusCallback;
};

} // namespace GPU::Window

#endif // EASYGPU_IWINDOW_PLATFORM_H
