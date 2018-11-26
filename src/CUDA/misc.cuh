#ifndef CPPNNET_MISC_CUH
#define CPPNNET_MISC_CUH

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__

#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>

#endif // __JETBRAINS_IDE__

#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>

typedef Eigen::MatrixXf Ematrix;
typedef Eigen::VectorXf Evector;
typedef Eigen::Map<Eigen::MatrixXf> Ematrixmap;
typedef Eigen::Map<Eigen::VectorXf> Evectormap;

#define HOSTDEVICE __host__ __device__

#include <stdio.h>

#define cudaerrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

struct kernel_parameters {
  int nnlayers;
  int niterations;
  int nsamtar;

};

__global__ void BackProp(float *weights, float *biases, float *samples, float *targets, float *senTa, float *sen,
                         int *dim, kernel_parameters params);

#endif // CPPNNET_MISC_CUH