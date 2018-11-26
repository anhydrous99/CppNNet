//
// Created by Armando Herrera on 11/20/2018.
//

#ifndef CPPNNET_CUMATRIX_CUH
#define CPPNNET_CUMATRIX_CUH

#include "misc.cuh"

template<class T>
class cumatrix {
  // types:
  typedef T &reference;
  typedef const T &const_reference;
  typedef size_t size_type;
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;

  // data:
  size_type N, M;
  pointer elemns;
  pointer d_elemns;
  bool in_device = false;

public:
  // Constructor & Desctructor
  HOSTDEVICE cumatrix(size_type n, size_type m) {
    elemns = new T[n * m];
    N = n;
    M = m;
  }

  __host__ cumatrix(std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> cont) {
    N = cont.size();
    M = cont[0].size();
    elemns = new T[N * M];
  }

  HOSTDEVICE cumatrix(pointer data_ptr, size_type n, size_type m) {
    elemns = data_ptr;
    N = n;
    M = m;
  }

  HOSTDEVICE ~cumatrix() {
    delete[] elemns;
#ifndef __CUDA_ARCH__
    if (in_device) release_device_data();
#endif
  }

  // capacity:
  HOSTDEVICE constexpr size_type size() const { return N * M; }

  HOSTDEVICE constexpr size_type rows() const { return N; }

  HOSTDEVICE constexpr size_type cols() const { return M; }

  // element access:
  HOSTDEVICE reference operator[](size_type i) { return elemns[i]; }

  HOSTDEVICE const_reference operator[](size_type i) const { return elemns[i]; }

  HOSTDEVICE reference operator()(size_type x, size_type y) { return elemns[x * M + y]; }

  HOSTDEVICE const_reference operator()(size_type x, size_type y) const { return elemns[x * M + y]; }

  HOSTDEVICE reference at(size_type i) { return elemns[i]; }

  HOSTDEVICE const_pointer cat(size_type i) const { return elemns[i]; }

  HOSTDEVICE reference at(size_type x, size_type y) { return elemns[x * M + y]; }

  HOSTDEVICE const_reference cat(size_type x, size_type y) const { return elemns[x * M + y]; }

  // direct access:
  HOSTDEVICE pointer data() { return elemns; }

  HOSTDEVICE const_pointer cdata() { return elemns; }

  // CONVERTED ACCESS:
  HOSTDEVICE Evectormap get_eigen_vector(size_type i) {
    return Evectormap(elemns[i * M], M);
  }

  // CUDA functions:
  __host__ pointer get_device_pointer(bool copy = true) {
    if (!in_device) {
      cudaerrchk(cudaMalloc((void **) &d_elemns, sizeof(T) * N * M));
      if (copy) cudaerrchk(cudaMemcpy(d_elemns, elemns, sizeof(T) * N * M, cudaMemcpyHostToDevice));
      in_device = true;
    }
    return d_elemns;
  }

  __host__ void refresh_from_device() {
    if (in_device) cudaerrchk(cudaMemcpy(elemns, d_elemns, sizeof(T) * N * M, cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_to_device() {
    if (in_device) cudaerrchk(cudaMemcpy(d_elemns, elemns, sizeof(T) * N * M, cudaMemcpyHostToDevice));
  }

  __host__ void release_device_data() {
    if (in_device) {
      cudaerrchk(cudaFree(d_elemns));
      in_device = false;
    }
  }
};

#endif // CPPNNET_CUMATRIX_CUH