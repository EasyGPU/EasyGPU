cmake_minimum_required(VERSION 3.21)

# =============================================================================
# Project Information
# =============================================================================
project(EasyGPU
    VERSION 0.1.0
    DESCRIPTION "A C++20 embedded DSL for GPU programming"
    LANGUAGES CXX C
)

# =============================================================================
# Options
# =============================================================================
option(EASYGPU_BUILD_EXAMPLES "Build example programs" ON)
option(EASYGPU_BUILD_TESTS "Build test suite" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Compiler Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories for the main library
include_directories(include third)

# =============================================================================
# Dependencies
# =============================================================================
# GLAD - OpenGL loader
add_library(GLAD "third/GLAD/glad.c")
target_include_directories(GLAD PUBLIC third)

set(EASYGPU_PLATFORM_LIBS opengl32)

# =============================================================================
# Core Library
# =============================================================================
add_library(EasyGPU STATIC
    # IR Builder
    include/IR/Builder/Builder.h
    source/IR/Builder/Builder.cpp
    include/IR/Builder/BuilderContext.h
    
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
    
    # IR Values
    include/IR/Value/Value.h
    source/IR/Value/Value.cpp
    include/IR/Value/Var.h
    source/IR/Value/Var.cpp
    include/IR/Value/VarArray.h
    source/IR/Value/VarArray.cpp
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
    include/IR/Value/TextureRef.h
    
    # Kernel
    include/Kernel/Kernel.h
    source/Kernel/Kernel.cpp
    include/Kernel/KernelBuildContext.h
    source/Kernel/KernelBuildContext.cpp
    include/Kernel/KernelProfiler.h
    source/Kernel/KernelProfiler.cpp
    
    # Flow Control
    include/Flow/IfFlow.h
    source/Flow/IfFlow.cpp
    include/Flow/ForFlow.h
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
    include/Runtime/Texture.h
    include/Runtime/Uniform.h
    include/Runtime/Context.h
    source/Runtime/Context.cpp
    include/Runtime/PixelFormat.h
    include/Runtime/ShaderException.h
    include/Runtime/ShaderUtils.h
    source/Runtime/ShaderUtils.cpp
    
    # Utilities
    include/Utility/Vec.h
    source/Utility/Vec.cpp
    include/Utility/Matrix.h
    source/Utility/Matrix.cpp
    include/Utility/Math.h
    source/Utility/Math.cpp
    include/Utility/Helpers.h
    include/Utility/Meta/StructMeta.h
    include/Utility/Meta/Std430Layout.h
    
    # Lazy Header
    include/GPU.h
        source/IR/Node/Load.cpp
)

target_compile_features(EasyGPU PUBLIC cxx_std_20)
target_link_libraries(EasyGPU PUBLIC GLAD ${EASYGPU_PLATFORM_LIBS})

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
    
    set(EASYGPU_TEST_SOURCES
        tests/TestVec.cpp
        tests/TestStruct.cpp
        tests/TestDSL.cpp
        tests/TestKernel1D.cpp
        tests/TestBuffer.cpp
        tests/TestTexture.cpp
        tests/TestFlowIf.cpp
        tests/TestFlowFor.cpp
        tests/TestFlowDoWhile.cpp
        tests/TestBreakContinue.cpp
        tests/TestMath.cpp
        tests/TestExpression.cpp
        tests/TestCallable.cpp
        tests/TestInspector.cpp
        tests/TestProfiler.cpp
        tests/TestDispatch.cpp
        tests/TestArray.cpp
        tests/TestUniform.cpp
    )
    
    foreach(test_source ${EASYGPU_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE EasyGPU)
        if(MSVC)
            target_compile_options(${test_name} PRIVATE /Zc:preprocessor)
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
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
    )
    
    set(EASYGPU_EXAMPLE_NAMES
        hello_world
        mandelbrot
        julia_set
        ray_tracing
        sdf_renderer
        volumetric_fog
    )
    
    list(LENGTH EASYGPU_EXAMPLE_SOURCES example_count)
    math(EXPR example_count "${example_count} - 1")
    
    foreach(i RANGE ${example_count})
        list(GET EASYGPU_EXAMPLE_SOURCES ${i} example_source)
        list(GET EASYGPU_EXAMPLE_NAMES ${i} example_name)
        add_executable(${example_name} ${example_source})
        target_link_libraries(${example_name} PRIVATE EasyGPU)
        if(MSVC)
            target_compile_options(${example_name} PRIVATE /Zc:preprocessor)
        endif()
    endforeach()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "EasyGPU Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${EASYGPU_BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${EASYGPU_BUILD_TESTS}")
message(STATUS "")
