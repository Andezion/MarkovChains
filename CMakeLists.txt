cmake_minimum_required(VERSION 3.20)

project(Markov LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

add_executable(MarkovApp
        main.cpp
        markov.cu
        markov.cuh
)

target_link_libraries(MarkovApp PRIVATE curand)

set_target_properties(MarkovApp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(MarkovTests
        tests.cpp
        markov.cu
        markov.cuh
)

target_link_libraries(MarkovTests PRIVATE curand)
set_target_properties(MarkovTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)