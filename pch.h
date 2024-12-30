//
// Created by Vasyl on 30.12.2024.
//

#ifndef PCH_H
#define PCH_H

#include <array>
#include <optional>
#include <vector>
#include <SDL2/SDL_video.h>
#include <vulkan/vulkan_core.h>
#define SDL_MAIN_HANDLED
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <vma/vk_mem_alloc.h>
#include "camera.h"

#include "SDL2/SDL_vulkan.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <vector>
#include "SDL2/SDL.h"
#include "shaders.h"
#include <unordered_map>

#include "engine.h"

#endif //PCH_H
