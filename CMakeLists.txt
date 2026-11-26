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
    src/ane_builder.mm
    src/dflash.mm
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
    MOMETAL_METALLIB_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_options(mometal PRIVATE
    -O3
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

# Optional build-time .metallib precompilation (eliminates ~1.24 s cold shader compile when MetalToolchain is installed)
execute_process(
    COMMAND xcrun metal --version
    RESULT_VARIABLE MOMETAL_HAS_METAL_RC
    OUTPUT_QUIET ERROR_QUIET
)
execute_process(
    COMMAND xcrun --find metallib
    RESULT_VARIABLE MOMETAL_HAS_METALLIB_RC
    OUTPUT_QUIET ERROR_QUIET
)

if(MOMETAL_HAS_METAL_RC EQUAL 0 AND MOMETAL_HAS_METALLIB_RC EQUAL 0)
    message(STATUS "mometal: Metal offline toolchain detected; precompiling .metallib shaders into ${CMAKE_CURRENT_BINARY_DIR}")
    set(MOMETAL_METALLIBS "")
    function(mometal_add_metallib TAG SRC_NAME STD_VER OUT_SUFFIX)
        set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SRC_NAME}")
        set(AIR_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TAG}_${OUT_SUFFIX}.air")
        set(LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TAG}_${OUT_SUFFIX}.metallib")
        add_custom_command(
            OUTPUT "${LIB_PATH}"
            COMMAND xcrun metal -c -std=${STD_VER} -ffast-math -O2 "${SRC_PATH}" -o "${AIR_PATH}"
            COMMAND xcrun metallib "${AIR_PATH}" -o "${LIB_PATH}"
            DEPENDS "${SRC_PATH}"
            COMMENT "Precompiling Metal shader ${SRC_NAME} (${STD_VER}) -> ${TAG}_${OUT_SUFFIX}.metallib"
            VERBATIM
        )
        list(APPEND MOMETAL_METALLIBS "${LIB_PATH}")
        set(MOMETAL_METALLIBS "${MOMETAL_METALLIBS}" PARENT_SCOPE)
    endfunction()

    mometal_add_metallib("common"   "common.metal"   "metal3.0" "m30")
    mometal_add_metallib("qwen38"   "qwen38.metal"   "metal3.0" "m30")
    mometal_add_metallib("dflash"   "dflash.metal"   "metal3.0" "m30")
    mometal_add_metallib("inkling2" "inkling2.metal" "metal3.0" "m30")
    mometal_add_metallib("vision"   "vision.metal"   "metal3.0" "m30")
    mometal_add_metallib("vision"   "vision.metal"   "metal3.1" "m31")

    add_custom_target(mometal_shaders ALL DEPENDS ${MOMETAL_METALLIBS})
    add_dependencies(mometal mometal_shaders)
else()
    message(STATUS "mometal: Metal offline toolchain unavailable (install via 'xcodebuild -downloadComponent MetalToolchain' to enable .metallib precompile); using runtime Metal shader cache")
endif()
