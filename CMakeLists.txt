cmake_minimum_required(VERSION 2.8.9)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.33.tar.gz"
    SHA1 "acbf4b9b77b5a0160adc1da1cdda35f64b14196a"
)

project(CppNNet LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

include_directories(include)

set(SOURCES src/Normalizer.cpp src/Neural_Layer.cpp src/Neural_Trainer.cpp src/main.cpp)

add_executable(CppNNet ${SOURCES})

target_link_libraries(CppNNet Eigen3::Eigen OpenMP::OpenMP_CXX)
