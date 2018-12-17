#include "misc.cuh"
#include "cumatrix.cuh"
#include "cuNetwork.cuh"

__global__ void BackProp(float *weights, float *biases, float *samples, float *targets, float *senTa, float *sen,
                         int *dim, kernel_parameters params) {
}