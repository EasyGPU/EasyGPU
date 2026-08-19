#include <Backend/VulkanBackend.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr int kSkipped = 77;

bool		  HasKhronosValidationLayer() {
	uint32_t layerCount = 0;
	if (vkEnumerateInstanceLayerProperties(&layerCount, nullptr) != VK_SUCCESS) {
		return false;
	}

	std::vector<VkLayerProperties> layers(layerCount);
	if (layerCount != 0 && vkEnumerateInstanceLayerProperties(&layerCount, layers.data()) != VK_SUCCESS) {
		return false;
	}

	return std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties &layer) {
		return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0;
	});
}

} // namespace

int main() {
	const char *validation = std::getenv("EASYGPU_ENABLE_VALIDATION");
	if (validation == nullptr || validation[0] == '\0' || std::strcmp(validation, "0") == 0) {
		std::cerr << "EASYGPU_ENABLE_VALIDATION must be enabled by the test wrapper\n";
		return 1;
	}
	if (!HasKhronosValidationLayer()) {
		std::cout << "VK_LAYER_KHRONOS_validation is unavailable; validation smoke test skipped\n";
		return kSkipped;
	}

	try {
		GPU::Backend::VulkanBackend backend;
		backend.Initialize();
		if (!backend.IsInitialized()) {
			std::cerr << "Vulkan backend did not report initialized state\n";
			return 1;
		}
		backend.Shutdown();
	} catch (const std::exception &error) {
		std::cerr << "Vulkan validation initialization failed: " << error.what() << '\n';
		return 1;
	}

	std::cout << "Vulkan validation initialization passed\n";
	return 0;
}
