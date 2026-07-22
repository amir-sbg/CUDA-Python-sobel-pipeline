cmake_minimum_required(VERSION 3.24)

project(cuda_saxpy_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_saxpy src/saxpy.cu)

set_target_properties(cuda_saxpy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
