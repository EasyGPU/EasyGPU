#include <Window/UIContext.h>

#include "Platform/GLFWWindowPlatform.h"

#ifdef EASYGPU_BACKEND_VULKAN
#include "Swapchain.h"
#include <Backend/VulkanBackend.h>
#include <Runtime/Context.h>
#endif

#include <imgui.h>
#include <imgui_impl_glfw.h>

#ifdef EASYGPU_BACKEND_VULKAN
#include <imgui_impl_vulkan.h>
#endif

#ifdef EASYGPU_BACKEND_OPENGL
#include <imgui_impl_opengl3.h>
#endif

#include <stdexcept>

namespace GPU::Window {

struct UIContext::Impl {
	AppWindow &window;

#ifdef EASYGPU_BACKEND_VULKAN
	GLFWWindowPlatform			*platform		= nullptr;
	Swapchain					*swapchain		= nullptr;
	GPU::Backend::VulkanBackend *backend		= nullptr;
	VkDescriptorPool			 descriptorPool = VK_NULL_HANDLE;
	bool						 frameActive	= false;
	bool						 drawDataReady	= false;

	explicit Impl(AppWindow &window_) : window(window_) {
		platform = dynamic_cast<GLFWWindowPlatform *>(window.Platform());
		if (!platform || !platform->GetSwapchain()) {
			throw std::runtime_error("UIContext requires a GLFW/Vulkan AppWindow");
		}
		swapchain						 = platform->GetSwapchain();
		backend							 = &GPU::Runtime::Context::GetBackend<GPU::Backend::VulkanBackend>();

		VkDescriptorPoolSize poolSizes[] = {
			{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
			{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
			{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
			{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
			{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
			{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
			{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
		};
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags		   = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.maxSets	   = 1000 * static_cast<uint32_t>(std::size(poolSizes));
		poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
		poolInfo.pPoolSizes	   = poolSizes;
		if (vkCreateDescriptorPool(backend->Device(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create ImGui descriptor pool");
		}

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui_ImplGlfw_InitForVulkan(platform->NativeWindow(), true);

		ImGui_ImplVulkan_InitInfo initInfo{};
		initInfo.Instance											 = backend->Instance();
		initInfo.PhysicalDevice										 = backend->PhysicalDevice();
		initInfo.Device												 = backend->Device();
		initInfo.QueueFamily										 = backend->QueueFamilyIndex();
		initInfo.Queue												 = backend->Queue();
		initInfo.PipelineCache										 = VK_NULL_HANDLE;
		initInfo.DescriptorPool										 = descriptorPool;
		initInfo.Subpass											 = 0;
		initInfo.MinImageCount										 = std::max(2u, swapchain->ImageCount());
		initInfo.ImageCount											 = swapchain->ImageCount();
		initInfo.MSAASamples										 = VK_SAMPLE_COUNT_1_BIT;
		initInfo.UseDynamicRendering								 = true;
		initInfo.PipelineRenderingCreateInfo.sType					 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		initInfo.PipelineRenderingCreateInfo.colorAttachmentCount	 = 1;
		VkFormat colorFormat										 = swapchain->Format();
		initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
		ImGui_ImplVulkan_Init(&initInfo);
		ImGui_ImplVulkan_CreateFontsTexture();
	}

	~Impl() {
		vkDeviceWaitIdle(backend->Device());
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		if (descriptorPool) {
			vkDestroyDescriptorPool(backend->Device(), descriptorPool, nullptr);
		}
	}

	void BeginFrame() {
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		frameActive	  = true;
		drawDataReady = false;
	}

	void EndFrame() {
		if (!frameActive) {
			return;
		}
		ImGui::Render();
		frameActive	  = false;
		drawDataReady = true;

		window.SetNextVulkanOverlay([this](VkCommandBuffer commandBuffer, uint32_t) {
			if (!drawDataReady) {
				return;
			}
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
			drawDataReady = false;
		});
	}
#elif defined(EASYGPU_BACKEND_OPENGL)
	GLFWWindowPlatform *platform	= nullptr;
	bool				frameActive = false;
	bool				drawDataReady = false;

	explicit Impl(AppWindow &window_) : window(window_) {
		platform = dynamic_cast<GLFWWindowPlatform *>(window.Platform());
		if (!platform || !platform->NativeWindow()) {
			throw std::runtime_error("UIContext requires a GLFW/OpenGL AppWindow");
		}

		glfwMakeContextCurrent(platform->NativeWindow());
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui_ImplGlfw_InitForOpenGL(platform->NativeWindow(), true);
		if (!ImGui_ImplOpenGL3_Init("#version 330 core")) {
			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext();
			glfwMakeContextCurrent(nullptr);
			throw std::runtime_error("Failed to initialize ImGui OpenGL backend");
		}
		glfwMakeContextCurrent(nullptr);
	}

	~Impl() {
		if (platform && platform->NativeWindow()) {
			glfwMakeContextCurrent(platform->NativeWindow());
		}
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		if (platform && platform->NativeWindow()) {
			glfwMakeContextCurrent(nullptr);
		}
	}

	void BeginFrame() {
		glfwMakeContextCurrent(platform->NativeWindow());
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		frameActive	  = true;
		drawDataReady = false;
		glfwMakeContextCurrent(nullptr);
	}

	void EndFrame() {
		if (!frameActive) {
			return;
		}
		glfwMakeContextCurrent(platform->NativeWindow());
		ImGui::Render();
		glfwMakeContextCurrent(nullptr);
		frameActive	  = false;
		drawDataReady = true;

		platform->SetOpenGLOverlay([this]() {
			if (!drawDataReady) {
				return;
			}
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			drawDataReady = false;
		});
	}
#else
	explicit Impl(AppWindow &window_) : window(window_) {
		throw std::runtime_error("UIContext requires the Vulkan or OpenGL backend");
	}
#endif
};

UIContext::UIContext(AppWindow &window) : _impl(std::make_unique<Impl>(window)) {
}

UIContext::~UIContext() = default;

void UIContext::BeginFrame() {
#if defined(EASYGPU_BACKEND_VULKAN) || defined(EASYGPU_BACKEND_OPENGL)
	_impl->BeginFrame();
#endif
}

void UIContext::EndFrame() {
#if defined(EASYGPU_BACKEND_VULKAN) || defined(EASYGPU_BACKEND_OPENGL)
	_impl->EndFrame();
#endif
}

void UIContext::Render(const std::function<void()> &uiFunc) {
	BeginFrame();
	if (uiFunc) {
		uiFunc();
	}
	EndFrame();
}

bool UIContext::WantCaptureKeyboard() const {
#if defined(EASYGPU_BACKEND_VULKAN) || defined(EASYGPU_BACKEND_OPENGL)
	return ImGui::GetIO().WantCaptureKeyboard;
#else
	return false;
#endif
}

bool UIContext::WantCaptureMouse() const {
#if defined(EASYGPU_BACKEND_VULKAN) || defined(EASYGPU_BACKEND_OPENGL)
	return ImGui::GetIO().WantCaptureMouse;
#else
	return false;
#endif
}

} // namespace GPU::Window
