#pragma once

/**
 * @file WindowConfig.h
 * @brief Window configuration structure
 */

#ifndef EASYGPU_WINDOW_CONFIG_H
#define EASYGPU_WINDOW_CONFIG_H

#include <cstdint>
#include <string>

namespace GPU::Window {

/**
 * @brief Configuration options for creating a window
 *
 * Usage:
 *   Window window({.width = 800, .height = 600, .title = "My App"});
 */
struct WindowConfig {
	uint32_t	width		   = 1280;
	uint32_t	height		   = 720;
	std::string title		   = "EasyGPU";
	bool		resizable	   = true;
	bool		visible		   = true;
	bool		vsync		   = true;
	bool		highDPI		   = true;
	bool		centerOnCreate = true;
};

} // namespace GPU::Window

#endif // EASYGPU_WINDOW_CONFIG_H
