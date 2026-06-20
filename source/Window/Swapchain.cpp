#include "Swapchain.h"

#ifdef EASYGPU_BACKEND_VULKAN

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace GPU::Window {
namespace {

void CheckVk(VkResult result, const char *operation) {
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string(operation) + " failed");
	}
}

} // namespace

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Create(const SwapchainConfig &config) {
	_config					 = config;
	_device					 = config.device;
	_vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
		vkGetDeviceProcAddr(_device, "vkCmdBeginRenderingKHR"));
	_vkCmdEndRenderingKHR =
		reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkGetDeviceProcAddr(_device, "vkCmdEndRenderingKHR"));
	if (!_vkCmdBeginRenderingKHR || !_vkCmdEndRenderingKHR) {
		throw std::runtime_error("Vulkan dynamic rendering is required for ImGui window overlays");
	}

	VkBool32 supportsPresent = VK_FALSE;
	vkGetPhysicalDeviceSurfaceSupportKHR(_config.physicalDevice, _config.queueFamilyIndex, _config.surface,
										 &supportsPresent);
	if (!supportsPresent) {
		throw std::runtime_error("Vulkan queue family does not support presenting to this surface");
	}

	CreateSwapchain();
	CreateCommandResources();
}

void Swapchain::Destroy() {
	if (!_device) {
		return;
	}
	vkDeviceWaitIdle(_device);
	DestroyStagingBuffer();
	DestroyCommandResources();
	DestroySwapchain();
	_device = VK_NULL_HANDLE;
}

void Swapchain::Recreate(uint32_t width, uint32_t height) {
	if (!_device) {
		return;
	}
	_config.width  = width;
	_config.height = height;
	if (width == 0 || height == 0) {
		_surfacePaused = true;
		vkDeviceWaitIdle(_device);
		DestroySwapchain();
		return;
	}
	_surfacePaused = false;
	vkDeviceWaitIdle(_device);
	DestroySwapchain();
	CreateSwapchain();
}

uint32_t Swapchain::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
	VkPhysicalDeviceMemoryProperties memoryProperties{};
	vkGetPhysicalDeviceMemoryProperties(_config.physicalDevice, &memoryProperties);
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((typeFilter & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

void Swapchain::CreateSwapchain() {
	VkSurfaceCapabilitiesKHR capabilities{};
	CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_config.physicalDevice, _config.surface, &capabilities),
			"vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	uint32_t formatCount = 0;
	CheckVk(vkGetPhysicalDeviceSurfaceFormatsKHR(_config.physicalDevice, _config.surface, &formatCount, nullptr),
			"vkGetPhysicalDeviceSurfaceFormatsKHR");
	std::vector<VkSurfaceFormatKHR> formats(formatCount);
	CheckVk(vkGetPhysicalDeviceSurfaceFormatsKHR(_config.physicalDevice, _config.surface, &formatCount, formats.data()),
			"vkGetPhysicalDeviceSurfaceFormatsKHR");

	uint32_t presentModeCount = 0;
	CheckVk(
		vkGetPhysicalDeviceSurfacePresentModesKHR(_config.physicalDevice, _config.surface, &presentModeCount, nullptr),
		"vkGetPhysicalDeviceSurfacePresentModesKHR");
	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	CheckVk(vkGetPhysicalDeviceSurfacePresentModesKHR(_config.physicalDevice, _config.surface, &presentModeCount,
													  presentModes.data()),
			"vkGetPhysicalDeviceSurfacePresentModesKHR");

	VkSurfaceFormatKHR surfaceFormat = formats.front();
	for (const auto &candidate : formats) {
		if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM && candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			surfaceFormat = candidate;
			break;
		}
	}

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if (!_config.vsync) {
		for (const auto candidate : presentModes) {
			if (candidate == VK_PRESENT_MODE_MAILBOX_KHR) {
				presentMode = candidate;
				break;
			}
		}
	}

	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		_extent = capabilities.currentExtent;
	} else {
		_extent.width  = std::clamp(std::max(_config.width, 1u), capabilities.minImageExtent.width,
									capabilities.maxImageExtent.width);
		_extent.height = std::clamp(std::max(_config.height, 1u), capabilities.minImageExtent.height,
									capabilities.maxImageExtent.height);
	}
	if (_extent.width == 0 || _extent.height == 0) {
		_surfacePaused = true;
		return;
	}
	_surfacePaused		= false;
	_format				= surfaceFormat.format;

	uint32_t imageCount = capabilities.minImageCount + 1;
	if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
		imageCount = capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType			= VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface			= _config.surface;
	createInfo.minImageCount	= imageCount;
	createInfo.imageFormat		= surfaceFormat.format;
	createInfo.imageColorSpace	= surfaceFormat.colorSpace;
	createInfo.imageExtent		= _extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage		= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.preTransform		= capabilities.currentTransform;
	createInfo.compositeAlpha	= VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode		= presentMode;
	createInfo.clipped			= VK_TRUE;
	createInfo.oldSwapchain		= VK_NULL_HANDLE;
	CheckVk(vkCreateSwapchainKHR(_device, &createInfo, nullptr, &_swapchain), "vkCreateSwapchainKHR");

	uint32_t actualImageCount = 0;
	CheckVk(vkGetSwapchainImagesKHR(_device, _swapchain, &actualImageCount, nullptr), "vkGetSwapchainImagesKHR");
	_images.resize(actualImageCount);
	CheckVk(vkGetSwapchainImagesKHR(_device, _swapchain, &actualImageCount, _images.data()), "vkGetSwapchainImagesKHR");

	_imageViews.resize(_images.size());
	for (size_t i = 0; i < _images.size(); ++i) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image							 = _images[i];
		viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format							 = _format;
		viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel	 = 0;
		viewInfo.subresourceRange.levelCount	 = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount	 = 1;
		CheckVk(vkCreateImageView(_device, &viewInfo, nullptr, &_imageViews[i]), "vkCreateImageView");
	}
}

void Swapchain::DestroySwapchain() {
	for (auto view : _imageViews) {
		vkDestroyImageView(_device, view, nullptr);
	}
	_imageViews.clear();
	_images.clear();
	if (_swapchain) {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
		_swapchain = VK_NULL_HANDLE;
	}
}

void Swapchain::CreateCommandResources() {
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = _config.queueFamilyIndex;
	poolInfo.flags			  = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	CheckVk(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool), "vkCreateCommandPool");

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool		 = _commandPool;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;
	CheckVk(vkAllocateCommandBuffers(_device, &allocInfo, &_commandBuffer), "vkAllocateCommandBuffers");

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	CheckVk(vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &_imageAvailable), "vkCreateSemaphore");
	CheckVk(vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &_renderFinished), "vkCreateSemaphore");

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	CheckVk(vkCreateFence(_device, &fenceInfo, nullptr, &_inFlightFence), "vkCreateFence");
}

void Swapchain::DestroyCommandResources() {
	if (_inFlightFence) {
		vkDestroyFence(_device, _inFlightFence, nullptr);
		_inFlightFence = VK_NULL_HANDLE;
	}
	if (_renderFinished) {
		vkDestroySemaphore(_device, _renderFinished, nullptr);
		_renderFinished = VK_NULL_HANDLE;
	}
	if (_imageAvailable) {
		vkDestroySemaphore(_device, _imageAvailable, nullptr);
		_imageAvailable = VK_NULL_HANDLE;
	}
	if (_commandPool) {
		vkDestroyCommandPool(_device, _commandPool, nullptr);
		_commandPool   = VK_NULL_HANDLE;
		_commandBuffer = VK_NULL_HANDLE;
	}
}

void Swapchain::BeginOverlayRendering(uint32_t imageIndex) {
	VkRenderingAttachmentInfoKHR colorAttachment{};
	colorAttachment.sType		= VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
	colorAttachment.imageView	= _imageViews[imageIndex];
	colorAttachment.imageLayout	= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachment.loadOp		= VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachment.storeOp		= VK_ATTACHMENT_STORE_OP_STORE;

	VkRenderingInfoKHR renderingInfo{};
	renderingInfo.sType					= VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
	renderingInfo.renderArea.offset		= {0, 0};
	renderingInfo.renderArea.extent		= _extent;
	renderingInfo.layerCount			= 1;
	renderingInfo.colorAttachmentCount	= 1;
	renderingInfo.pColorAttachments		= &colorAttachment;
	_vkCmdBeginRenderingKHR(_commandBuffer, &renderingInfo);
}

void Swapchain::EndOverlayRendering() {
	_vkCmdEndRenderingKHR(_commandBuffer);
}

void Swapchain::EnsureStagingBuffer(size_t requiredSize) {
	if (_stagingBuffer && _stagingSize >= requiredSize) {
		return;
	}
	DestroyStagingBuffer();
	_stagingSize = requiredSize;

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size		   = _stagingSize;
	bufferInfo.usage	   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	CheckVk(vkCreateBuffer(_device, &bufferInfo, nullptr, &_stagingBuffer), "vkCreateBuffer");

	VkMemoryRequirements requirements{};
	vkGetBufferMemoryRequirements(_device, _stagingBuffer, &requirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize  = requirements.size;
	allocInfo.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
																				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	CheckVk(vkAllocateMemory(_device, &allocInfo, nullptr, &_stagingMemory), "vkAllocateMemory");
	CheckVk(vkBindBufferMemory(_device, _stagingBuffer, _stagingMemory, 0), "vkBindBufferMemory");
}

void Swapchain::DestroyStagingBuffer() {
	if (_stagingBuffer) {
		vkDestroyBuffer(_device, _stagingBuffer, nullptr);
		_stagingBuffer = VK_NULL_HANDLE;
	}
	if (_stagingMemory) {
		vkFreeMemory(_device, _stagingMemory, nullptr);
		_stagingMemory = VK_NULL_HANDLE;
	}
	_stagingSize = 0;
}

uint32_t Swapchain::AcquireImage() {
	if (_surfacePaused || !_swapchain || _extent.width == 0 || _extent.height == 0) {
		throw std::runtime_error("Cannot acquire swapchain image while surface extent is zero");
	}
	CheckVk(vkWaitForFences(_device, 1, &_inFlightFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
	CheckVk(vkResetFences(_device, 1, &_inFlightFence), "vkResetFences");

	uint32_t imageIndex = 0;
	VkResult result =
		vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, _imageAvailable, VK_NULL_HANDLE, &imageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		Recreate(_config.width, _config.height);
		return AcquireImage();
	}
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		CheckVk(result, "vkAcquireNextImageKHR");
	}
	return imageIndex;
}

void Swapchain::BeginCommands() {
	CheckVk(vkResetCommandBuffer(_commandBuffer, 0), "vkResetCommandBuffer");
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	CheckVk(vkBeginCommandBuffer(_commandBuffer, &beginInfo), "vkBeginCommandBuffer");
}

void Swapchain::SubmitAndPresent(uint32_t imageIndex) {
	CheckVk(vkEndCommandBuffer(_commandBuffer), "vkEndCommandBuffer");

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo		 submitInfo{};
	submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount	= 1;
	submitInfo.pWaitSemaphores		= &_imageAvailable;
	submitInfo.pWaitDstStageMask	= &waitStage;
	submitInfo.commandBufferCount	= 1;
	submitInfo.pCommandBuffers		= &_commandBuffer;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores	= &_renderFinished;
	CheckVk(vkQueueSubmit(_config.queue, 1, &submitInfo, _inFlightFence), "vkQueueSubmit");

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType			   = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores	   = &_renderFinished;
	presentInfo.swapchainCount	   = 1;
	presentInfo.pSwapchains		   = &_swapchain;
	presentInfo.pImageIndices	   = &imageIndex;

	VkResult result				   = vkQueuePresentKHR(_config.queue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		Recreate(_config.width, _config.height);
	} else {
		CheckVk(result, "vkQueuePresentKHR");
	}
}

void Swapchain::TransitionImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
								VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage,
								VkPipelineStageFlags dstStage) const {
	VkImageMemoryBarrier barrier{};
	barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout						= oldLayout;
	barrier.newLayout						= newLayout;
	barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.image							= image;
	barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel	= 0;
	barrier.subresourceRange.levelCount		= 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount		= 1;
	barrier.srcAccessMask					= srcAccess;
	barrier.dstAccessMask					= dstAccess;
	vkCmdPipelineBarrier(_commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Swapchain::CopyPixelsToImage(const uint32_t *pixels, uint32_t width, uint32_t height, uint32_t imageIndex) {
	const size_t bytes = static_cast<size_t>(width) * height * sizeof(uint32_t);
	EnsureStagingBuffer(bytes);

	void *mapped = nullptr;
	CheckVk(vkMapMemory(_device, _stagingMemory, 0, bytes, 0, &mapped), "vkMapMemory");
	if (_format == VK_FORMAT_B8G8R8A8_UNORM || _format == VK_FORMAT_B8G8R8A8_SRGB) {
		auto *dst = static_cast<uint32_t *>(mapped);
		for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
			uint32_t rgba = pixels[i];
			uint32_t r	  = rgba & 0x000000FFu;
			uint32_t g	  = rgba & 0x0000FF00u;
			uint32_t b	  = rgba & 0x00FF0000u;
			uint32_t a	  = rgba & 0xFF000000u;
			dst[i]		  = a | (r << 16) | g | (b >> 16);
		}
	} else {
		std::memcpy(mapped, pixels, bytes);
	}
	vkUnmapMemory(_device, _stagingMemory);

	TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
					VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferImageCopy region{};
	region.bufferOffset					   = 0;
	region.bufferRowLength				   = width;
	region.bufferImageHeight			   = height;
	region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel	   = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount	   = 1;
	region.imageOffset					   = {0, 0, 0};
	region.imageExtent					   = {std::min(width, _extent.width), std::min(height, _extent.height), 1};
	vkCmdCopyBufferToImage(_commandBuffer, _stagingBuffer, _images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
						   &region);
}

void Swapchain::BlitTextureToImage(GPU::Backend::VulkanBackend &backend, GPU::Backend::TextureHandle texture,
								   uint32_t imageIndex) {
	auto native = backend.GetNativeTextureInfo(texture);
	if (!native.image || native.depth != 1) {
		throw std::runtime_error("TexturePresenter requires a valid 2D Vulkan texture");
	}

	TransitionImage(native.image, native.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_ACCESS_TRANSFER_READ_BIT,
					VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
					VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkImageBlit blit{};
	blit.srcSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.srcSubresource.mipLevel	   = 0;
	blit.srcSubresource.baseArrayLayer = 0;
	blit.srcSubresource.layerCount	   = 1;
	blit.srcOffsets[0]				   = {0, 0, 0};
	blit.srcOffsets[1]				   = {static_cast<int32_t>(native.width), static_cast<int32_t>(native.height), 1};
	blit.dstSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.dstSubresource.mipLevel	   = 0;
	blit.dstSubresource.baseArrayLayer = 0;
	blit.dstSubresource.layerCount	   = 1;
	blit.dstOffsets[0]				   = {0, 0, 0};
	blit.dstOffsets[1]				   = {static_cast<int32_t>(_extent.width), static_cast<int32_t>(_extent.height), 1};
	vkCmdBlitImage(_commandBuffer, native.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _images[imageIndex],
				   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

	backend.SetNativeTextureLayout(texture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
								   VK_ACCESS_TRANSFER_READ_BIT);
}

void Swapchain::PresentPixels(const uint32_t *pixels, uint32_t width, uint32_t height, const OverlayCallback &overlay) {
	if (!pixels || width == 0 || height == 0 || _surfacePaused || !_swapchain) {
		return;
	}
	uint32_t imageIndex = AcquireImage();
	BeginCommands();
	CopyPixelsToImage(pixels, width, height, imageIndex);
	if (overlay) {
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
						VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		BeginOverlayRendering(imageIndex);
		overlay(_commandBuffer, imageIndex);
		EndOverlayRendering();
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
						VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	} else {
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
						VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	}
	SubmitAndPresent(imageIndex);
}

void Swapchain::PresentTexture(GPU::Backend::VulkanBackend &backend, GPU::Backend::TextureHandle texture,
							   const OverlayCallback &overlay) {
	if (_surfacePaused || !_swapchain) {
		return;
	}
	uint32_t imageIndex = AcquireImage();
	BeginCommands();
	BlitTextureToImage(backend, texture, imageIndex);
	if (overlay) {
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
						VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		BeginOverlayRendering(imageIndex);
		overlay(_commandBuffer, imageIndex);
		EndOverlayRendering();
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
						VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	} else {
		TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
						VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	}
	SubmitAndPresent(imageIndex);
}

void Swapchain::PresentOverlayOnly(const OverlayCallback &overlay) {
	if (!overlay || _surfacePaused || !_swapchain) {
		return;
	}
	uint32_t imageIndex = AcquireImage();
	BeginCommands();
	TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	BeginOverlayRendering(imageIndex);
	overlay(_commandBuffer, imageIndex);
	EndOverlayRendering();
	TransitionImage(_images[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	SubmitAndPresent(imageIndex);
}

} // namespace GPU::Window

#endif // EASYGPU_BACKEND_VULKAN
