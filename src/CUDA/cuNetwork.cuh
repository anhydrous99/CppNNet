//
// Created by Armando Herrera on 11/20/2018.
//

#ifndef CPPNNET_CUNETWORK_CUH
#define CPPNNET_CUNETWORK_CUH

#include "misc.cuh"
#include <math.h>

template<class T>
class cuNetwork {
  // types:
  typedef T &reference;
  typedef const T &const_reference;
  typedef size_t size_type;
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;

  // data:
  size_type nnlayers;
  size_type nweights;
  size_type nbiases;
  size_type *dim;
  pointer weights;
  pointer biases;
  pointer d_weights;
  pointer d_biases;
  bool in_device = false;

public:
  // constructors
  HOSTDEVICE cuNetwork(std::vector<Eigen::MatrixXf> Weights, std::vector<Eigen::VectorXf> Biases) {
    nnlayers = Weights.size();
    dim = T[nnlayers * 2];
    for (size_type i = 0; i < nnlayers; i++) {
      dim[i * 2] = Weights[i].rows();
      dim[i * 2 + 1] = Weights[i].cols();
    }

    nweights = 0;
    nbiases = 0;
    for (size_type i = 0; i < nnlayers; i++) {
      nweights += dim[i * 2] * dim[i * 2 + 1];
      nbiases += dim[i * 2];
    }

    weights = new T[nweights];
    biases = new T[biases];
    size_type weightn = 0;
    size_type biasesn = 0;
    for (size_type m = 0; m < nnlayers; m++) {
      for (size_type i = 0; i < dim[m * 2]; i++) {
        for (size_type j = 0; j < dim[m * 2 + 1]; j++) {
          weights[weightn] = Weights[m](i, j);
          weightn++;
        }
        biases[biasesn] = Biases[m][i];
        biasesn++;
      }
    }
  }

  HOSTDEVICE cuNetwork(size_type nn_layers, size_type *dims, pointer Weights, pointer Biases) {
    nnlayers = nn_layers;
    dim = dims;
    weights = Weights;
    biases = Biases;
    nweights = 0;
    nbiases = 0;
    for (size_type i = 0; i < nnlayers; i++) {
      nweights += dim[i * 2] * dim[i * 2 + 1];
      nbiases += dim[i * 2];
    }
  }

  HOSTDEVICE ~cuNetwork() {
    delete[] dim;
    delete[] weights;
    delete[] biases;
    if (in_device) release_device_data();
  }

  // capacity
  HOSTDEVICE constexpr size_type size() const {
    return nweights + nbiases;
  }

  HOSTDEVICE constexpr size_type weights_size() const {
    return nweights;
  }

  HOSTDEVICE constexpr size_type biases_size() const {
    return biases;
  }

  HOSTDEVICE size_type weights_size(size_type i) const {
    return dim[i * 2] * dim[i * 2 + 1];
  }

  HOSTDEVICE size_type biases_size(size_type i) const {
    return dim[i * 2];
  }

  HOSTDEVICE size_type weights_size(size_type i, size_type j) const {
    size_type sum = 0;
    size_type u = min(i, j), v = max(i, j);
    for (size_type m = u; m < v; m++)
      sum += dim[m * 2] * dim[m * 2 + 1];
    return sum;
  }

  HOSTDEVICE size_type biases_size(size_type i, size_type j) const {
    size_type sum = 0;
    size_type u = min(i, j), v = max(i, j);
    for (size_type m = u; m < v; m++)
      sum += dim[m * 2];
    return sum;
  }

  HOSTDEVICE size_type weights_rows(size_type m) const { return dim[m * 2]; }

  HOSTDEVICE size_type weights_cols(size_type m) const { return dim[m * 2 + 1]; }

  // element access:
  HOSTDEVICE reference operator()(size_type m, size_type i, size_type j) { return weights_accessor(m, i, j); }

  HOSTDEVICE const_reference
  operator()(size_type m, size_type i, size_type j) const { return weights_accessor(m, i, j); }

  HOSTDEVICE reference operator()(size_type m, size_type i) { return biases_accessor(m, i); }

  HOSTDEVICE const_reference operator()(size_type m, size_type i) const { return biases_accessor(m, i); }

  HOSTDEVICE reference get_weight(size_type m, size_type i, size_type j) { return weights_accessor(m, i, j); }

  HOSTDEVICE const_reference
  cget_weight(size_type m, size_type i, size_type j) const { return weights_accessor(m, i, j); }

  HOSTDEVICE reference get_bias(size_type m, size_type i) { return biases_accessor(m, i); }

  HOSTDEVICE const_reference cget_bias(size_type m, size_type i) const { return biases_accessor(m, i); }

  // direct access:
  HOSTDEVICE pointer weight_data() { return weights; }

  HOSTDEVICE const_pointer cweight_data() const { return weights; }

  HOSTDEVICE pointer bias_data() { return biases; }

  HOSTDEVICE const_pointer cbias_data() const { return biases; }

  HOSTDEVICE pointer weight_device_data() { return (in_device) ? d_weights : nullptr; }

  HOSTDEVICE const_pointer cweight_device_data() const { return (in_device) ? d_weights : nullptr; }

  HOSTDEVICE pointer bias_device_data() { return (in_device) ? d_biases : nullptr; }

  HOSTDEVICE const_pointer cbias_device_data() const { return (in_device) ? d_biases : nullptr; }

  // converted access:
  HOSTDEVICE Ematrixmap get_eigen_weights(size_type m) {
    return Ematrixmap(&weights[weights_size(0, m)], weights_rows(m), weights_cols(m));
  }

  HOSTDEVICE Evectormap get_eigen_vector(size_type m) {
    return Evectormap(&biases[biases_size(0, m)], biases_size(m));
  }

  // CUDA functions
  __host__ void get_device_pointers(pointer &weights_ptr, pointer &biases_ptr, bool copy = true) {
    cudaerrchk(cudaMalloc((void **) &weights_ptr, sizeof(T) * nweights));
    cudaerrchk(cudaMalloc((void **) &biases_ptr, sizeof(T) * nbiases));
    if (copy) {
      refresh_weights_to_device();
      refresh_biases_to_device();
    }
    d_weights = weights_ptr;
    d_biases = biases_ptr;
    in_device = true;
  }

  __host__ void refresh_weights_from_device() {
    cudaerrchk(cudaMemcpy(weights, d_weights, sizeof(T) * nweights, cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_weights_to_device() {
    cudaerrchk(cudaMemcpy(d_weights, weights, sizeof(T) * nweights, cudaMemcpyHostToDevice));
  }

  __host__ void refresh_biases_from_device() {
    cudaerrchk(cudaMemcpy(biases, d_biases, sizeof(T) * nbiases, cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_biases_to_device() {
    cudaerrchk(cudaMemcpy(d_biases, biases, sizeof(T) * nbiases, cudaMemcpyHostToDevice));
  }

  __host__ void release_device_data() {
    if (in_device) {
      cudaerrchk(cudaFree(d_weights));
      cudaerrchk(cudaFree(d_biases));
      in_device = false;
    }
  }

  // utility functions
private:
  HOSTDEVICE inline reference weights_accessor(size_type layer, size_type i, size_type j) {
    return weights[weights_size(0, layer) + i * weights_cols(layer) + j];
  }

  HOSTDEVICE inline reference biases_accessor(size_type layer, size_type i) {
    return biases[biases_size(0, layer) + i];
  }
};

#define // CPPNNET_CUNETWORK_CUH