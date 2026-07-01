cmake_minimum_required(VERSION 3.21)

# =============================================================================
# Project Information
# =============================================================================
project(EasyGPU
    VERSION 2.0.0
    DESCRIPTION "A C++20 embedded DSL for GPU programming"
    LANGUAGES CXX C
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# =============================================================================
# Options
# =============================================================================
option(EASYGPU_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(EASYGPU_BUILD_TESTS "Build test suite" ${PROJECT_IS_TOP_LEVEL})
option(EASYGPU_BUILD_FRAGMENT_TESTER "Build fragment kernel tester with EasyX" OFF)
option(EASYGPU_BUILD_WINDOW "Build window component for interactive visualization" ON)
option(EASYGPU_ENABLE_SHADER_CACHE "Enable shader binary caching for faster startup" ON)
option(EASYGPU_ENABLE_SPIRV_OPT "Optimize Vulkan SPIR-V with SPIRV-Tools opt passes" ON)
option(EASYGPU_ENABLE_SPIRV_CROSS_GLSL "Expose optimized Vulkan SPIR-V as GLSL through SPIRV-Cross" ON)
set(EASYGPU_SHADER_CACHE_DIR "" CACHE STRING "Directory for shader cache (empty = use default)")
option(EASYGPU_BUILD_WINDOW_EXAMPLES "Build window examples" ${PROJECT_IS_TOP_LEVEL})
set(EASYGPU_BACKEND "Vulkan" CACHE STRING "Backend API to use (OpenGL or Vulkan)")
set_property(CACHE EASYGPU_BACKEND PROPERTY STRINGS OpenGL Vulkan)

if(NOT EASYGPU_BACKEND STREQUAL "OpenGL" AND NOT EASYGPU_BACKEND STREQUAL "Vulkan")
    message(FATAL_ERROR "EASYGPU_BACKEND must be either OpenGL or Vulkan, got: ${EASYGPU_BACKEND}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Compiler Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # Treat all source files as UTF-8 to prevent C4819 warnings on non-ASCII characters
    add_compile_options(/utf-8)
    # Optionally suppress C4819 entirely for downstream users on non-UTF-8 code pages
    add_compile_options(/wd4819)
endif()

if(MSVC AND EASYGPU_BACKEND STREQUAL "Vulkan")
    # The Vulkan SDK ships glslang/SPIRV static libraries built against the
    # release DLL CRT and STL iterator debug level 0. Match that ABI even for
    # Debug builds, otherwise MSVC reports LNK2038 mismatches.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
endif()

# =============================================================================
# Dependencies
# =============================================================================
# GLAD - OpenGL loader (OpenGL backend only)
if(EASYGPU_BACKEND STREQUAL "OpenGL")
    add_library(GLAD "third/GLAD/glad.c")
    target_include_directories(GLAD PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

if(EASYGPU_BACKEND STREQUAL "Vulkan")
    find_package(Vulkan REQUIRED)
endif()

# Platform-specific libraries
if(WIN32)
    set(EASYGPU_PLATFORM_LIBS crypt32)
    if(EASYGPU_BACKEND STREQUAL "OpenGL")
        list(APPEND EASYGPU_PLATFORM_LIBS opengl32)
    endif()
elseif(APPLE)
    find_package(OpenSSL REQUIRED)
    set(EASYGPU_PLATFORM_LIBS OpenSSL::Crypto)
    if(EASYGPU_BACKEND STREQUAL "OpenGL")
        message(FATAL_ERROR "OpenGL backend is not supported on macOS. Use -DEASYGPU_BACKEND=Vulkan on macOS.")
    endif()
elseif(UNIX AND NOT APPLE)
    if(EASYGPU_BACKEND STREQUAL "OpenGL")
        find_package(OpenGL REQUIRED)
    endif()
    find_package(X11 REQUIRED)
    find_package(OpenSSL REQUIRED)
    set(EASYGPU_PLATFORM_LIBS ${X11_LIBRARIES} OpenSSL::Crypto)
    if(EASYGPU_BACKEND STREQUAL "OpenGL")
        list(APPEND EASYGPU_PLATFORM_LIBS ${OPENGL_LIBRARIES})
    endif()
endif()

# =============================================================================
# Window Component (GLFW + ImGui)
# =============================================================================
if(EASYGPU_BUILD_WINDOW)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    if(UNIX AND NOT APPLE)
        set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    endif()
    if(EASYGPU_BACKEND STREQUAL "Vulkan")
        set(GLFW_VULKAN_STATIC OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(third/glfw)

    set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third/imgui)
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    )

    if(EASYGPU_BACKEND STREQUAL "Vulkan")
        target_sources(imgui PRIVATE ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
        target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
    elseif(EASYGPU_BACKEND STREQUAL "OpenGL")
        target_sources(imgui PRIVATE ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
    endif()

    target_include_directories(imgui PUBLIC
        $<BUILD_INTERFACE:${IMGUI_DIR}>
        $<BUILD_INTERFACE:${IMGUI_DIR}/backends>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/imgui>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/imgui/backends>
    )
    target_link_libraries(imgui PUBLIC glfw)
    target_compile_features(imgui PUBLIC cxx_std_17)
    
    # Create EasyGPUWindow library
    add_library(EasyGPUWindow STATIC
        include/Window/WindowConfig.h
        include/Window/WindowEvents.h
        include/Window/Input.h
        include/Window/PixelBuffer.h
        include/Window/AppWindow.h
        include/Window/UIContext.h
        include/Window/TexturePresenter.h
        include/Window/TexturePresenter.inl
        source/Window/PixelBuffer.cpp
        source/Window/Window.cpp
        source/Window/TexturePresenter.cpp
        source/Window/UIContext.cpp
        source/Window/Swapchain.h
        source/Window/Swapchain.cpp
        source/Window/Platform/IWindowPlatform.h
        source/Window/Platform/MiniFBWindowPlatform.h
        source/Window/Platform/GLFWWindowPlatform.h
        source/Window/Platform/GLFWWindowPlatform.cpp
    )
    add_library(EasyGPU::Window ALIAS EasyGPUWindow)
    
    target_compile_features(EasyGPUWindow PUBLIC cxx_std_20)
    target_compile_definitions(EasyGPUWindow PUBLIC EASYGPU_BUILD_WINDOW)
    target_link_libraries(EasyGPUWindow PUBLIC glfw imgui)
    target_include_directories(EasyGPUWindow PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third/glfw/include>
        $<INSTALL_INTERFACE:include>
    )
    
    if(MSVC)
        target_compile_options(EasyGPUWindow PUBLIC /Zc:preprocessor)
    endif()
endif()

if(EASYGPU_BACKEND STREQUAL "Vulkan")
    set(EASYGPU_VULKAN_SDK_HINTS)
    set(EASYGPU_GLSLANG_CONFIG_HINTS)
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND EASYGPU_VULKAN_SDK_HINTS
            "$ENV{VULKAN_SDK}/lib"
            "$ENV{VULKAN_SDK}/Lib"
            "$ENV{VULKAN_SDK}/Lib32"
        )
        list(APPEND EASYGPU_GLSLANG_CONFIG_HINTS
            "$ENV{VULKAN_SDK}/lib/cmake/glslang"
            "$ENV{VULKAN_SDK}/Lib/cmake/glslang"
        )
    endif()

    get_filename_component(EASYGPU_VULKAN_LIBRARY_DIR "${Vulkan_LIBRARY}" DIRECTORY)
    list(APPEND EASYGPU_GLSLANG_CONFIG_HINTS
        "${EASYGPU_VULKAN_LIBRARY_DIR}/cmake/glslang"
        "${EASYGPU_VULKAN_LIBRARY_DIR}/cmake/vulkan/glslang"
    )
    find_package(glslang CONFIG QUIET HINTS ${EASYGPU_GLSLANG_CONFIG_HINTS})

    if(TARGET glslang::SPIRV AND TARGET glslang::glslang-default-resource-limits)
        set(EASYGPU_USE_GLSLANG_PACKAGE ON)

        find_package(SPIRV-Tools CONFIG QUIET)
        if(TARGET SPIRV-Tools-static AND NOT TARGET SPIRV-Tools)
            add_library(SPIRV-Tools INTERFACE IMPORTED)
            set_property(TARGET SPIRV-Tools PROPERTY INTERFACE_LINK_LIBRARIES SPIRV-Tools-static)
        endif()
        find_package(SPIRV-Tools-opt CONFIG QUIET)
        find_package(SPIRV-Tools-link CONFIG QUIET)

        set(EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS)
        if(TARGET SPIRV-Tools-link)
            list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS SPIRV-Tools-link)
        else()
            find_library(SPIRV_TOOLS_LINK_LIBRARY SPIRV-Tools-link HINTS ${EASYGPU_VULKAN_SDK_HINTS})
            if(SPIRV_TOOLS_LINK_LIBRARY)
                list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS ${SPIRV_TOOLS_LINK_LIBRARY})
            endif()
        endif()
        if(TARGET SPIRV-Tools-opt)
            list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS SPIRV-Tools-opt)
        else()
            find_library(SPIRV_TOOLS_OPT_LIBRARY SPIRV-Tools-opt HINTS ${EASYGPU_VULKAN_SDK_HINTS})
            if(SPIRV_TOOLS_OPT_LIBRARY)
                list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS ${SPIRV_TOOLS_OPT_LIBRARY})
            elseif(EASYGPU_ENABLE_SPIRV_OPT)
                message(FATAL_ERROR "EASYGPU_ENABLE_SPIRV_OPT requires SPIRV-Tools-opt")
            endif()
        endif()
        if(TARGET SPIRV-Tools-static)
            list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS SPIRV-Tools-static)
        elseif(TARGET SPIRV-Tools)
            list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS SPIRV-Tools)
        else()
            find_library(SPIRV_TOOLS_LIBRARY SPIRV-Tools HINTS ${EASYGPU_VULKAN_SDK_HINTS})
            if(SPIRV_TOOLS_LIBRARY)
                list(APPEND EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS ${SPIRV_TOOLS_LIBRARY})
            endif()
        endif()

        set(EASYGPU_VULKAN_LIBS
            Vulkan::Vulkan
            glslang::glslang
            glslang::SPIRV
            glslang::glslang-default-resource-limits
            ${EASYGPU_PACKAGE_SPIRV_TOOLS_LIBS}
        )
    else()
        set(EASYGPU_USE_GLSLANG_PACKAGE OFF)
        find_library(GLSLANG_LIBRARY glslang HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(GLSLANG_SPIRV_LIBRARY SPIRV HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(GLSLANG_RESOURCE_LIMITS_LIBRARY glslang-default-resource-limits HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(GLSLANG_MACHINEINDEPENDENT_LIBRARY MachineIndependent HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(GLSLANG_OSDEPENDENT_LIBRARY OSDependent HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(GLSLANG_GENERICCODEGEN_LIBRARY GenericCodeGen HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(SPIRV_TOOLS_LIBRARY SPIRV-Tools HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(SPIRV_TOOLS_OPT_LIBRARY SPIRV-Tools-opt HINTS ${EASYGPU_VULKAN_SDK_HINTS})

        set(EASYGPU_REQUIRED_VULKAN_LIBS
            GLSLANG_LIBRARY
            GLSLANG_SPIRV_LIBRARY
            GLSLANG_RESOURCE_LIMITS_LIBRARY
            GLSLANG_MACHINEINDEPENDENT_LIBRARY
            GLSLANG_OSDEPENDENT_LIBRARY
            GLSLANG_GENERICCODEGEN_LIBRARY
            SPIRV_TOOLS_LIBRARY
        )
        if(EASYGPU_ENABLE_SPIRV_OPT)
            list(APPEND EASYGPU_REQUIRED_VULKAN_LIBS SPIRV_TOOLS_OPT_LIBRARY)
        endif()
        foreach(required_lib ${EASYGPU_REQUIRED_VULKAN_LIBS})
            if(NOT ${required_lib})
                message(FATAL_ERROR "Missing required Vulkan/glslang dependency: ${required_lib}")
            endif()
        endforeach()

        set(EASYGPU_SPIRV_OPT_LIBS)
        if(EASYGPU_ENABLE_SPIRV_OPT)
            list(APPEND EASYGPU_SPIRV_OPT_LIBS ${SPIRV_TOOLS_OPT_LIBRARY})
        endif()

        set(EASYGPU_VULKAN_LIBS
            Vulkan::Vulkan
            ${GLSLANG_LIBRARY}
            ${GLSLANG_SPIRV_LIBRARY}
            ${GLSLANG_RESOURCE_LIMITS_LIBRARY}
            ${GLSLANG_MACHINEINDEPENDENT_LIBRARY}
            ${GLSLANG_OSDEPENDENT_LIBRARY}
            ${GLSLANG_GENERICCODEGEN_LIBRARY}
            ${SPIRV_TOOLS_LIBRARY}
            ${EASYGPU_SPIRV_OPT_LIBS}
        )
    endif()

    if(EASYGPU_ENABLE_SPIRV_CROSS_GLSL)
        find_library(SPIRV_CROSS_CORE_LIBRARY spirv-cross-core HINTS ${EASYGPU_VULKAN_SDK_HINTS})
        find_library(SPIRV_CROSS_GLSL_LIBRARY spirv-cross-glsl HINTS ${EASYGPU_VULKAN_SDK_HINTS})

        if(NOT SPIRV_CROSS_CORE_LIBRARY OR NOT SPIRV_CROSS_GLSL_LIBRARY)
            message(FATAL_ERROR "EASYGPU_ENABLE_SPIRV_CROSS_GLSL requires spirv-cross-core and spirv-cross-glsl")
        endif()

        list(APPEND EASYGPU_VULKAN_LIBS
            ${SPIRV_CROSS_GLSL_LIBRARY}
            ${SPIRV_CROSS_CORE_LIBRARY}
        )
    endif()
endif()

if(EASYGPU_BACKEND STREQUAL "OpenGL")
    set(EASYGPU_BACKEND_SOURCES
        include/Backend/OpenGLBackend.h
        source/Backend/OpenGLBackend.cpp
    )
elseif(EASYGPU_BACKEND STREQUAL "Vulkan")
    set(EASYGPU_BACKEND_SOURCES
        include/Backend/VulkanBackend.h
        source/Backend/VulkanBackend.cpp
    )
endif()

# =============================================================================
# Core Library
# =============================================================================
add_library(EasyGPU STATIC
    # Backend
    include/Backend/Backend.h
    source/Backend/Backend.cpp
    ${EASYGPU_BACKEND_SOURCES}
    
    # IR Builder
    include/IR/Builder/Builder.h
    source/IR/Builder/Builder.cpp
    include/IR/Builder/BuilderContext.h
    include/IR/Module.h
    source/IR/Module.cpp
    
    # IR Nodes
    include/IR/Node/Node.h
    include/IR/Node/LocalVariable.h
    source/IR/Node/LocalVariable.cpp
    include/IR/Node/LocalVariableArray.h
    source/IR/Node/LocalVariableArray.cpp
    include/IR/Node/Load.h
    include/IR/Node/LoadLocalVariable.h
    source/IR/Node/LoadLocalVariable.cpp
    include/IR/Node/LoadLocalArray.h
    source/IR/Node/LoadLocalArray.cpp
    include/IR/Node/LoadUniform.h
    source/IR/Node/LoadUniform.cpp
    include/IR/Node/Store.h
    source/IR/Node/Store.cpp
    include/IR/Node/Operation.h
    source/IR/Node/Operation.cpp
    include/IR/Node/CallInst.h
    source/IR/Node/CallInst.cpp
    include/IR/Node/Call.h
    source/IR/Node/Call.cpp
    include/IR/Node/ArrayAccess.h
    source/IR/Node/ArrayAccessNode.cpp
    include/IR/Node/CompoundAssignment.h
    source/IR/Node/CompoundAssignment.cpp
    include/IR/Node/Increment.h
    source/IR/Node/Increment.cpp
    include/IR/Node/MemberAccess.h
    source/IR/Node/MemberAccess.cpp
    source/IR/Node/If.cpp
    include/IR/Node/For.h
    source/IR/Node/For.cpp
    include/IR/Node/While.h
    source/IR/Node/While.cpp
    include/IR/Node/DoWhile.h
    source/IR/Node/DoWhile.cpp
    include/IR/Node/Break.h
    source/IR/Node/Break.cpp
    include/IR/Node/Continue.h
    source/IR/Node/Continue.cpp
    include/IR/Node/Return.h
    source/IR/Node/Return.cpp
    include/IR/Node/RawCode.h
    source/IR/Node/RawCode.cpp
    include/IR/Node/Ternary.h
    source/IR/Node/Ternary.cpp
    include/IR/Node/SharedMemory.h
    source/IR/Node/SharedMemory.cpp
    include/IR/Node/AtomicOp.h
    source/IR/Node/AtomicOp.cpp
    include/IR/Node/Barrier.h
    source/IR/Node/Barrier.cpp
    include/IR/Node/TextureLoad.h
    source/IR/Node/TextureLoad.cpp
    include/IR/Node/TextureStore.h
    source/IR/Node/TextureStore.cpp
    include/IR/Node/TextureSample.h
    source/IR/Node/TextureSample.cpp
    
    # IR Values
    include/IR/Value/Value.h
    source/IR/Value/Value.cpp
    include/IR/Value/Var.h
    source/IR/Value/Var.cpp
    include/IR/Value/VarArray.h
    source/IR/Value/VarArray.cpp
    include/IR/Value/SideEffectToken.h
    source/IR/Value/SideEffectToken.cpp
    include/IR/Value/VarVector.h
    include/IR/Value/VarIVector.h
    include/IR/Value/VarMatrix.h
    include/IR/Value/VarStruct.h
    include/IR/Value/Expr.h
    source/IR/Value/Expr.cpp
    include/IR/Value/ExprVector.h
    include/IR/Value/ExprIVector.h
    include/IR/Value/ExprMatrix.h
    include/IR/Value/BufferRef.h
    include/IR/Value/SharedMemory.h
    include/IR/Value/TextureRef.h
    include/IR/Value/TextureSampler.h
    
    # Kernel
    include/Kernel/Kernel.h
    source/Kernel/Kernel.cpp
    include/Kernel/KernelBuildContext.h
    source/Kernel/KernelBuildContext.cpp
    include/Kernel/KernelProfiler.h
    source/Kernel/KernelProfiler.cpp
    include/Kernel/ShaderCache.h
    source/Kernel/ShaderCache.cpp
    
    # Fragment Kernel (Windows only)
    include/Kernel/FragmentKernel.h
    source/Kernel/FragmentKernel.cpp
    include/Kernel/FragmentBuildContext.h
    source/Kernel/FragmentBuildContext.cpp
    include/Kernel/WindowAttachment.h
    source/Kernel/WindowAttachment.cpp
	    source/Kernel/GraphicsBuildContext.cpp
	    source/Kernel/GraphicsPipeline.cpp
	    source/Kernel/FragmentShader.cpp
    
    # Flow Control
    include/Flow/IfFlow.h
    source/Flow/IfFlow.cpp
    include/Flow/ForFlow.h
    source/Flow/ForFlow.cpp
    include/Flow/WhileFlow.h
    include/Flow/DoWhileFlow.h
    include/Flow/BreakFlow.h
    include/Flow/ContinueFlow.h
    include/Flow/ReturnFlow.h
    include/Flow/CodeCollectContext.h
    source/Flow/CodeCollectContext.cpp
    
    # Callable
    include/Callable/Callable.h
    
    # Runtime
    include/Runtime/Buffer.h
    include/Runtime/BufferSlot.h
    source/Runtime/BufferSlot.cpp
    include/Runtime/Texture.h
    include/Runtime/TextureSlot.h
    include/Runtime/Uniform.h
    include/Runtime/UniformBuffer.h
    include/Runtime/Context.h
    source/Runtime/Context.cpp
    include/Runtime/PixelFormat.h
    include/Runtime/ShaderException.h
    include/Runtime/ShaderUtils.h
    source/Runtime/ShaderUtils.cpp
    include/Runtime/GLStateCache.h
    source/Runtime/GLStateCache.cpp
    
    # Utilities
    include/Utility/Vec.h
    source/Utility/Vec.cpp
    include/Utility/Matrix.h
    source/Utility/Matrix.cpp
    include/Utility/Math.h
    source/Utility/Math.cpp
    include/Utility/Helpers.h
    include/Utility/Unref.h
    include/Utility/Atomic.h
    include/Utility/Parallel.h
    include/Utility/ThreadIndex.h
    include/Utility/Meta/StructMeta.h
    include/Utility/Meta/Std430Layout.h
    
    # Automatic Differentiation
    include/AD/TapeEntry.h
    include/AD/GradientTape.h
    source/AD/GradientTape.cpp
    include/AD/AdjointTable.h
    source/AD/AdjointTable.cpp
    include/AD/AdjointGenerator.h
    source/AD/AdjointGenerator.cpp
    include/AD/AdjointInspector.h
    include/AD/ADCore.h

    # Neural Network
    include/NN/NN.h
    include/NN/Tensor.h
    include/NN/Layers.h
    include/NN/Optimizer.h
    source/NN/Optimizer.cpp
    include/NN/Loss.h
    include/NN/Normalization.h
    include/NN/Embedding.h
    include/NN/Attention.h
    include/NN/Transformer.h
    include/NN/Checkpoint.h
    source/NN/Checkpoint.cpp
    include/NN/FusedMLP.h
    include/NN/FusedMLPTraining.h

    # Benchmark
    include/Benchmark/Benchmark.h
    source/Benchmark/Benchmark.cpp

    # Lazy Header
    include/GPU.h
    source/IR/Node/Load.cpp
)
add_library(EasyGPU::EasyGPU ALIAS EasyGPU)
set_target_properties(EasyGPU PROPERTIES EXPORT_NAME EasyGPU)
if(TARGET EasyGPUWindow)
    set_target_properties(EasyGPUWindow PROPERTIES EXPORT_NAME Window)
endif()

target_compile_features(EasyGPU PUBLIC cxx_std_20)
if(EASYGPU_BACKEND STREQUAL "OpenGL")
    target_link_libraries(EasyGPU PUBLIC GLAD ${EASYGPU_PLATFORM_LIBS})
else()
    target_link_libraries(EasyGPU PUBLIC ${EASYGPU_PLATFORM_LIBS})
endif()

if(EASYGPU_BACKEND STREQUAL "Vulkan")
    target_compile_definitions(EasyGPU PUBLIC EASYGPU_BACKEND_VULKAN)
    if(EASYGPU_ENABLE_SPIRV_OPT)
        target_compile_definitions(EasyGPU PRIVATE EASYGPU_SPIRV_OPT_ENABLED)
    endif()
    if(EASYGPU_ENABLE_SPIRV_CROSS_GLSL)
        target_compile_definitions(EasyGPU PRIVATE EASYGPU_SPIRV_CROSS_GLSL_ENABLED)
    endif()
    target_include_directories(EasyGPU PUBLIC $<BUILD_INTERFACE:${Vulkan_INCLUDE_DIR}>)
    target_link_libraries(EasyGPU PUBLIC
        "$<BUILD_INTERFACE:${EASYGPU_VULKAN_LIBS}>"
        "$<INSTALL_INTERFACE:Vulkan::Vulkan;EasyGPU::VulkanDependencies>"
    )
else()
    target_compile_definitions(EasyGPU PUBLIC EASYGPU_BACKEND_OPENGL)
    if(EASYGPU_BUILD_WINDOW)
        target_compile_definitions(EasyGPU PUBLIC EASYGPU_BUILD_WINDOW)
    endif()
endif()

if(TARGET EasyGPUWindow)
    target_link_libraries(EasyGPUWindow PUBLIC EasyGPU)
endif()

# Shader cache configuration
if(EASYGPU_ENABLE_SHADER_CACHE)
    target_compile_definitions(EasyGPU PUBLIC EASYGPU_SHADER_CACHE_ENABLED)
    if(NOT EASYGPU_SHADER_CACHE_DIR STREQUAL "")
        target_compile_definitions(EasyGPU PUBLIC EASYGPU_SHADER_CACHE_DIR="${EASYGPU_SHADER_CACHE_DIR}")
    endif()
endif()

# MSVC must use the conforming preprocessor for tuple-style variadic macros
# used by EASYGPU_STRUCT(..., (type,name), ...).
if(MSVC)
    target_compile_options(EasyGPU PUBLIC /Zc:preprocessor)
endif()

# Set include directories for users of the library
target_include_directories(EasyGPU PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third>
    $<INSTALL_INTERFACE:include>
)

# =============================================================================
# Tests
# =============================================================================
if(EASYGPU_BUILD_TESTS)
    enable_testing()

    file(GLOB EASYGPU_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/Test*.cpp")
    list(FILTER EASYGPU_TEST_SOURCES EXCLUDE REGEX "/TestFragmentKernel\\.cpp$")
    
    foreach(test_source ${EASYGPU_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE EasyGPU)
        if(MSVC)
            target_compile_options(${test_name} PRIVATE /Zc:preprocessor /UNDEBUG /Od)
        else()
            target_compile_options(${test_name} PRIVATE -UNDEBUG -O0)
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 120 LABELS "EasyGPU")
    endforeach()
endif()

# =============================================================================
# Examples
# =============================================================================
if(EASYGPU_BUILD_EXAMPLES)
    set(EASYGPU_EXAMPLE_SOURCES
        examples/hello_world/main.cpp
        examples/mandelbrot/main.cpp
        examples/julia_set/main.cpp
        examples/ray_tracing/main.cpp
        examples/sdf_renderer/main.cpp
        examples/volumetric_fog/main.cpp
        examples/parallel_reduction/main.cpp
        examples/histogram/main.cpp
        examples/spirv_opt_inspection/main.cpp
        examples/image_processing/color_filter/main.cpp
        examples/image_processing/gaussian_blur/main.cpp
        examples/image_processing/box_filter/main.cpp
        examples/ad_linear_regression/main.cpp
        examples/ad_transformer/main.cpp
        examples/ad_gpt_poet_demo/main.cpp
        examples/ad_gpt_demo/main.cpp
    )

    set(EASYGPU_EXAMPLE_NAMES
        hello_world
        mandelbrot
        julia_set
        ray_tracing
        sdf_renderer
        volumetric_fog
        parallel_reduction
        histogram
        spirv_opt_inspection
        color_filter
        gaussian_blur
        box_filter
        ad_linear_regression
        ad_transformer
        ad_gpt_poet_demo
        ad_gpt_demo
    )
    
    list(LENGTH EASYGPU_EXAMPLE_SOURCES example_count)
    math(EXPR example_count "${example_count} - 1")
    
    foreach(i RANGE ${example_count})
        list(GET EASYGPU_EXAMPLE_SOURCES ${i} example_source)
        list(GET EASYGPU_EXAMPLE_NAMES ${i} example_name)
        add_executable(${example_name} ${example_source})
        target_link_libraries(${example_name} PRIVATE EasyGPU)
        if(MSVC)
            target_compile_options(${example_name} PRIVATE /Zc:preprocessor /bigobj)
        endif()
    endforeach()
    
endif()

# =============================================================================
# Installation and CMake package export
# =============================================================================
set(EASYGPU_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/EasyGPU")

install(TARGETS EasyGPU
    EXPORT EasyGPUTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET GLAD)
    install(TARGETS GLAD EXPORT EasyGPUTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(TARGET EasyGPUWindow)
    install(TARGETS EasyGPUWindow glfw imgui
        EXPORT EasyGPUTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY third/GLAD/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GLAD FILES_MATCHING PATTERN "*.h")
if(TARGET EasyGPUWindow)
    install(DIRECTORY third/glfw/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY third/imgui/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/imgui
        FILES_MATCHING PATTERN "*.h"
    )
endif()

configure_package_config_file(
    cmake/EasyGPUConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/EasyGPUConfig.cmake"
    INSTALL_DESTINATION "${EASYGPU_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/EasyGPUConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT EasyGPUTargets
    FILE EasyGPUTargets.cmake
    NAMESPACE EasyGPU::
    DESTINATION "${EASYGPU_INSTALL_CMAKEDIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/EasyGPUConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/EasyGPUConfigVersion.cmake"
    DESTINATION "${EASYGPU_INSTALL_CMAKEDIR}"
)

# =============================================================================
# Fragment Kernel Tester (Windows + EasyX)
# =============================================================================
if(EASYGPU_BUILD_FRAGMENT_TESTER)
    if(NOT WIN32)
        message(WARNING "Fragment Kernel Tester is only supported on Windows")
    else()
        # EasyX is typically installed as a system library on Windows
        # or can be provided via EASYGPU_EASYX_PATH
        set(EASYGPU_EASYX_PATH "" CACHE PATH "Path to EasyX library")
        
        if(EASYGPU_EASYX_PATH)
            include_directories(${EASYGPU_EASYX_PATH}/include)
            link_directories(${EASYGPU_EASYX_PATH}/lib)
        endif()
        
        add_executable(TestFragmentKernel WIN32 tests/TestFragmentKernel.cpp)
        target_link_libraries(TestFragmentKernel PRIVATE EasyGPU)
        
        if(MSVC)
            target_compile_options(TestFragmentKernel PRIVATE /Zc:preprocessor)
        endif()
        
        message(STATUS "  Build Fragment Kernel Tester: ON")
    endif()
endif()

# =============================================================================
# Window Examples
# =============================================================================
if(EASYGPU_BUILD_WINDOW AND EASYGPU_BUILD_WINDOW_EXAMPLES)
    add_executable(window_hello examples/window_hello/main.cpp)
    target_link_libraries(window_hello PRIVATE EasyGPUWindow)
    
    add_executable(window_pixels examples/window_pixels/main.cpp)
    target_link_libraries(window_pixels PRIVATE EasyGPUWindow)
    
    if(MSVC)
        target_compile_options(window_hello PRIVATE /Zc:preprocessor)
        target_compile_options(window_pixels PRIVATE /Zc:preprocessor)
    endif()
    
    # GPU compute + window example (requires EasyGPU)
    if(TARGET EasyGPU)
        add_executable(window_compute examples/window_compute/main.cpp)
        target_link_libraries(window_compute PRIVATE EasyGPUWindow)
        add_executable(window_imgui_lab examples/window_imgui_lab/main.cpp)
        target_link_libraries(window_imgui_lab PRIVATE EasyGPUWindow)
        if(MSVC)
            target_compile_options(window_compute PRIVATE /Zc:preprocessor)
            target_compile_options(window_imgui_lab PRIVATE /Zc:preprocessor)
        endif()
    endif()
    
    # Real-time Cornell Box GI (requires EasyGPU + EasyGPUWindow)
    if(TARGET EasyGPU AND TARGET EasyGPUWindow)
        add_executable(cornell_box_gi examples/cornell_box_gi/main.cpp)
        target_link_libraries(cornell_box_gi PRIVATE EasyGPUWindow)
        if(MSVC)
            target_compile_options(cornell_box_gi PRIVATE /Zc:preprocessor)
        endif()
    endif()

    # OBJ Model Viewer — rasterization pipeline demo (requires EasyGPU + EasyGPUWindow)
    if(TARGET EasyGPU AND TARGET EasyGPUWindow)
        add_executable(obj_renderer examples/obj_renderer/main.cpp)
        target_link_libraries(obj_renderer PRIVATE EasyGPUWindow)
        if(MSVC)
            target_compile_options(obj_renderer PRIVATE /Zc:preprocessor)
        endif()
    endif()

    # Sponza Atrium — textured 3D model (requires EasyGPU + EasyGPUWindow)
    if(TARGET EasyGPU AND TARGET EasyGPUWindow)
        add_executable(sponza_renderer examples/sponza_renderer/main.cpp)
        target_link_libraries(sponza_renderer PRIVATE EasyGPUWindow)
        if(MSVC)
            target_compile_options(sponza_renderer PRIVATE /Zc:preprocessor)
        endif()
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "EasyGPU Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Backend: ${EASYGPU_BACKEND}")
message(STATUS "  Build Examples: ${EASYGPU_BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${EASYGPU_BUILD_TESTS}")
message(STATUS "  Build Fragment Kernel Tester: ${EASYGPU_BUILD_FRAGMENT_TESTER}")
message(STATUS "  Build Window: ${EASYGPU_BUILD_WINDOW}")
message(STATUS "  Shader Cache: ${EASYGPU_ENABLE_SHADER_CACHE}")
if(EASYGPU_BACKEND STREQUAL "Vulkan")
    message(STATUS "  SPIR-V Optimization: ${EASYGPU_ENABLE_SPIRV_OPT}")
    message(STATUS "  SPIR-V Cross GLSL: ${EASYGPU_ENABLE_SPIRV_CROSS_GLSL}")
endif()
if(EASYGPU_BUILD_WINDOW)
    message(STATUS "  Build Window Examples: ${EASYGPU_BUILD_WINDOW_EXAMPLES}")
endif()
message(STATUS "")
