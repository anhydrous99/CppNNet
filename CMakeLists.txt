cmake_minimum_required(VERSION 3.9)

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif ()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://s3.amazonaws.com/files.themadphysicist.com/hunter-0.23.48.tar.gz"
        SHA1 "32710f444638f25aa0a5aa26baef1284a69a1eca"
)

project(CppNNet)
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
endif (NOT CMAKE_BUILD_TYPE)

option(BUILD_TEST "Build the Tests" ON)
option(USE_CUDA "Build Cuda Components" ON)

#set(CMAKE_CXX_FLAGS_RELEASE "-g ${CMAKE_CXX_FLAGS_RELEASE}")

hunter_add_package(ZLIB)
hunter_add_package(Eigen)
hunter_add_package(CURL)
hunter_add_package(RapidJSON)
find_package(ZLIB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

if (USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(hasCUDA TRUE)
        if (NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif ()
    endif ()
endif ()

include_directories(include)

add_subdirectory(src)

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif ()
