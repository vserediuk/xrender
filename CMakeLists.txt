cmake_minimum_required(VERSION 3.29)
project(xrender)

set(CMAKE_CXX_STANDARD 20)

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")

file(COPY ${MODELS_DIR} DESTINATION ${CMAKE_BINARY_DIR})

set(TEXTURES_DIR "${CMAKE_SOURCE_DIR}/textures")

file(COPY ${TEXTURES_DIR} DESTINATION ${CMAKE_BINARY_DIR})

find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found.")
endif()

get_filename_component(Vulkan_LIB_DIR "${Vulkan_LIBRARIES}" DIRECTORY)

if(NOT Vulkan_LIB_DIR)
    message(FATAL_ERROR "Could not determine Vulkan_LIB_DIR")
endif()

include_directories(${Vulkan_INCLUDE_DIRS} third-party)

set(SPIRV_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/shaders_spv")
set(SHADER_INPUT_DIR "${CMAKE_SOURCE_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})
message(STATUS "SPIR-V output directory: ${SPIRV_OUTPUT_DIR}")

file(GLOB SHADER_FILES "shaders/*.frag" "shaders/*.vert" "shaders/*.comp")
message(STATUS "Found shaders: ${SHADER_FILES}")

set(ALL_SHADER_OUTPUT_FILES)

# Try to find the GLSLangValidator program more reliably
find_program(GLSLANG_VALIDATOR NAMES glslangValidator.exe PATHS ${Vulkan_LIB_DIR}/../Bin)

if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found")
endif()

message(STATUS "Using GLSLangValidator at: ${GLSLANG_VALIDATOR}")

foreach(shader ${SHADER_FILES})
    get_filename_component(shader_filename ${shader} NAME)
    get_filename_component(shader_base_name ${shader_filename} NAME_WE)
    get_filename_component(shader_extension ${shader_filename} EXT)

    # Define the output file path for the shader
    set(output_file "${SPIRV_OUTPUT_DIR}/${shader_base_name}${shader_extension}.spv")
    message(STATUS "Compiling shader: ${shader} to ${output_file}")

    add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${GLSLANG_VALIDATOR} -V ${shader} -o ${output_file}
            DEPENDS ${shader}
            COMMENT "Compiling shader ${shader} to SPIR-V"
    )


    list(APPEND ALL_SHADER_OUTPUT_FILES ${output_file})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${ALL_SHADER_OUTPUT_FILES})

add_executable(xrender main.cpp engine.cpp engine.h
        camera.cpp
        camera.h)

target_link_libraries(xrender Vulkan::Vulkan ${Vulkan_LIB_DIR}/SDL2.lib)

add_dependencies(xrender Shaders)
