#pragma once

#ifndef EASYGPU_WINDOW_SWAPCHAIN_H
#define EASYGPU_WINDOW_SWAPCHAIN_H

#ifdef EASYGPU_BACKEND_VULKAN

#include <Backend/VulkanBackend.h>

#include <functional>
#include <vector>
#include <vulkan/vulkan.h>

namespace GPU::Window {

struct SwapchainConfig {
	VkInstance		 instance		  = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice	  = VK_NULL_HANDLE;
	VkDevice		 device			  = VK_NULL_HANDLE;
	VkSurfaceKHR	 surface		  = VK_NULL_HANDLE;
	uint32_t		 queueFamilyIndex = 0;
	VkQueue			 queue			  = VK_NULL_HANDLE;
	uint32_t		 width			  = 0;
	uint32_t		 height			  = 0;
	bool			 vsync			  = true;
};

class Swapchain {
public:
	using OverlayCallback = std::function<void(VkCommandBuffer, uint32_t)>;

	Swapchain()			  = default;
	~Swapchain();

	Swapchain(const Swapchain &)			= delete;
	Swapchain &operator=(const Swapchain &) = delete;
	Swapchain(Swapchain &&)					= delete;
	Swapchain &operator=(Swapchain &&)		= delete;

	void	   Create(const SwapchainConfig &config);
	void	   Destroy();
	void	   Recreate(uint32_t width, uint32_t height);

	void PresentPixels(const uint32_t *pixels, uint32_t width, uint32_t height, const OverlayCallback &overlay = {});
	void PresentTexture(GPU::Backend::VulkanBackend &backend, GPU::Backend::TextureHandle texture,
						const OverlayCallback &overlay = {});
	void PresentOverlayOnly(const OverlayCallback &overlay);

	[[nodiscard]] VkFormat Format() const {
		return _format;
	}
	[[nodiscard]] VkExtent2D Extent() const {
		return _extent;
	}
	[[nodiscard]] uint32_t ImageCount() const {
		return static_cast<uint32_t>(_images.size());
	}
	[[nodiscard]] VkImage Image(uint32_t i) const {
		return _images[i];
	}
	[[nodiscard]] VkImageView ImageView(uint32_t i) const {
		return _imageViews[i];
	}
	[[nodiscard]] VkCommandBuffer CurrentCommandBuffer() const {
		return _commandBuffer;
	}

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	void	 CreateSwapchain();
	void	 DestroySwapchain();
	void	 CreateCommandResources();
	void	 DestroyCommandResources();
	void	 BeginOverlayRendering(uint32_t imageIndex);
	void	 EndOverlayRendering();
	void	 EnsureStagingBuffer(size_t requiredSize);
	void	 DestroyStagingBuffer();
	uint32_t AcquireImage();
	void	 BeginCommands();
	void	 SubmitAndPresent(uint32_t imageIndex);
	void TransitionImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccess,
						 VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) const;
	void CopyPixelsToImage(const uint32_t *pixels, uint32_t width, uint32_t height, uint32_t imageIndex);
	void BlitTextureToImage(GPU::Backend::VulkanBackend &backend, GPU::Backend::TextureHandle texture,
							uint32_t imageIndex);

private:
	SwapchainConfig			 _config{};
	VkDevice				 _device	= VK_NULL_HANDLE;
	VkSwapchainKHR			 _swapchain = VK_NULL_HANDLE;
	bool					 _surfacePaused = false;
	VkFormat				 _format	= VK_FORMAT_B8G8R8A8_UNORM;
	VkExtent2D				 _extent{};
	std::vector<VkImage>	 _images;
	std::vector<VkImageView> _imageViews;
	VkCommandPool			 _commandPool	 = VK_NULL_HANDLE;
	VkCommandBuffer			 _commandBuffer	 = VK_NULL_HANDLE;
	VkSemaphore				 _imageAvailable = VK_NULL_HANDLE;
	VkSemaphore				 _renderFinished = VK_NULL_HANDLE;
	VkFence					 _inFlightFence	 = VK_NULL_HANDLE;
	VkBuffer				 _stagingBuffer	 = VK_NULL_HANDLE;
	VkDeviceMemory			 _stagingMemory	 = VK_NULL_HANDLE;
	size_t					 _stagingSize	 = 0;
	PFN_vkCmdBeginRenderingKHR _vkCmdBeginRenderingKHR = nullptr;
	PFN_vkCmdEndRenderingKHR	 _vkCmdEndRenderingKHR	 = nullptr;
};

} // namespace GPU::Window

#endif // EASYGPU_BACKEND_VULKAN
#endif // EASYGPU_WINDOW_SWAPCHAIN_H
