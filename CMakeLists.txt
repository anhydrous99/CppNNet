cmake_minimum_required(VERSION 2.8.9)

cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0074 NEW)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.33.tar.gz"
        SHA1 "acbf4b9b77b5a0160adc1da1cdda35f64b14196a"
)

project(CppNNet)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

#set(CMAKE_CXX_FLAGS_RELEASE "-g ${CMAKE_CXX_FLAGS_RELEASE}")

hunter_add_package(ZLIB)
hunter_add_package(Eigen)
hunter_add_package(CURL)
find_package(ZLIB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

include_directories(include)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
