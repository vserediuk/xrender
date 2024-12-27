//
// Created by Vasyl on 26.12.2024.
//

#ifndef ENGINE_H
#define ENGINE_H
#include <array>
#include <optional>
#include <vector>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <vma/vk_mem_alloc.h>

const int MAX_FRAMES_IN_FLIGHT = 2;


struct AllocatedBuffer {
    VkBuffer _buffer;
    VmaAllocation _allocation;
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

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

    void cleanupSwapChain();

    void initAllocator();

    void recreateSwapChain();

    void createVertexBuffer();

    void drawFrame();

    void createSyncObjects();

    void createCommandBuffer();

    void recordCommandBuffer(VkCommandBuffer buffer, uint32_t imageIndex);

    void createCommandPool();

private:
    VmaAllocator allocator;
    AllocatedBuffer vertexBuffer;
    uint32_t currentFrame = 0;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkCommandBuffer> commandBuffers;
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
