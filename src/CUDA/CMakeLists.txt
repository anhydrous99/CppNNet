set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --expt-relaxed-constexpr --expt-extended-lambda")

include_directories(${EIGEN3_INCLUDE_DIRS})
add_library(CppNNet_CUDA OBJECT CUDA_Neural_Trainer.cu)

set_target_properties(CppNNet_CUDA PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SPARABLE_COMPILATION ON
        )