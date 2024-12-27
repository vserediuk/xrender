//
// Created by Vasyl on 26.12.2024.
//

#ifndef ENGINE_H
#define ENGINE_H
#include <optional>
#include <vector>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan_core.h>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME
    };

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class VKEngine {
public:
    void run();

    void drawFrame();

    void createSyncObjects();

    void createCommandBuffer();

    void recordCommandBuffer(VkCommandBuffer buffer, uint32_t imageIndex);

    void createCommandPool();

private:
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;
    VkCommandBuffer commandBuffer;
    VkCommandPool commandPool;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkImage> swapChainImages;
    VkPipeline graphicsPipeline;
    SDL_Window* window;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkQueue graphicsQueue;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkSwapchainKHR swapChain;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPipelineLayout pipelineLayout;

    void initWindow();

    void setupDebugMessenger();

    VkShaderModule createShaderModule(const uint32_t source[], size_t sourceSize);

    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                       const VkAllocationCallbacks *pAllocator);

    void createGraphicsPipeline();

    void initVulkan();

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    void mainLoop();

    void cleanup();

    void createSurface();

    void createVulkanInstance();

    static VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);

    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                          const VkAllocationCallbacks *pAllocator,
                                          VkDebugUtilsMessengerEXT *pDebugMessenger);

    std::vector<const char *> getRequiredExtensions();

    bool checkValidationLayerSupport();

    void createImageViews();

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    void createSwapChain();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    bool isDeviceSuitable(VkPhysicalDevice device);

    void createLogicalDevice();

    void pickPhysicalDevice();
};

#endif //ENGINE_H
