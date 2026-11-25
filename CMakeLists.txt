cmake_minimum_required(VERSION 3.24)
project(mometal LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(mometal
    src/safetensors.mm
    src/metal_ctx.mm
    src/ane_mlp.mm
    src/ane_share.mm
    src/vision.mm
    src/tokenizer.mm
    src/sampler.cpp
    src/qwen38.mm
    src/inkling2.mm
    src/server.cpp
    src/main.mm
)

target_include_directories(mometal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(mometal PRIVATE
    MOMETAL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

target_compile_options(mometal PRIVATE
    -O3
    -ffast-math
    -fno-math-errno
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-unused-function
    -fobjc-arc
)

target_link_libraries(mometal PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework ImageIO"
    "-framework CoreML"
    "-framework CoreVideo"
    "-framework IOSurface"
)
