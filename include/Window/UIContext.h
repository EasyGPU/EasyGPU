#pragma once

#ifndef EASYGPU_WINDOW_UI_CONTEXT_H
#define EASYGPU_WINDOW_UI_CONTEXT_H

#include <Window/AppWindow.h>

#include <functional>
#include <memory>

namespace GPU::Window {

class UIContext {
public:
	explicit UIContext(AppWindow &window);
	~UIContext();

	UIContext(const UIContext &)			   = delete;
	UIContext &operator=(const UIContext &)	   = delete;
	UIContext(UIContext &&)					   = delete;
	UIContext		  &operator=(UIContext &&) = delete;

	void			   BeginFrame();
	void			   EndFrame();
	void			   Render(const std::function<void()> &uiFunc);

	[[nodiscard]] bool WantCaptureKeyboard() const;
	[[nodiscard]] bool WantCaptureMouse() const;

private:
	struct Impl;
	std::unique_ptr<Impl> _impl;
};

} // namespace GPU::Window

#endif // EASYGPU_WINDOW_UI_CONTEXT_H
