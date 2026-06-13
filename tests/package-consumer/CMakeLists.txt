cmake_minimum_required(VERSION 3.21)

project(EasyGPUPackageConsumer LANGUAGES CXX)

find_package(EasyGPU CONFIG REQUIRED)

add_executable(easygpu-package-consumer main.cpp)
target_link_libraries(easygpu-package-consumer PRIVATE EasyGPU::EasyGPU)
