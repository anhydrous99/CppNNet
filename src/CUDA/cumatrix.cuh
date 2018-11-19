#ifndef CPPNNET_CUMATRIX_CUH
#define CPPNNET_CUMATRIX_CUH

#include "misc.cuh"

#include <thrust/swap.h>
#include <thrust/random.h>

template<class T>
struct cumatrix {
  // types:
  typedef cumatrix self;
  typedef self &selfref;
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *iterator;
  typedef const value_type *const_iterator;
  typedef int size_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;

  pointer elemns;
  pointer d_elemns;
  bool in_device = false;
  bool dodelete = true;
  size_type N;
  size_type M;

  // Constructor & Destructor
  HOSTDEVICE cumatrix(size_type n, size_type m) {
    N = n;
    M = m;
    elemns = new T[N * M];
  }

  HOSTDEVICE cumatrix(pointer &data, size_type n, size_type m, bool do_delete = true) {
    elemns = data;
    dodelete = do_delete;
    N = n;
    M = m;
  }

  HOSTDEVICE ~cumatrix() {
    if (dodelete) delete[] elemns;
  }

  // iterators
  HOSTDEVICE iterator begin() noexcept { return iterator(data()); }

  HOSTDEVICE const_iterator cbegin() const noexcept { return const_iterator(data()); }

  HOSTDEVICE iterator end() noexcept { return iterator(data() + N); }

  HOSTDEVICE const_iterator cend() const noexcept { return const_iterator(data() + N); }

  // Capacity
  HOSTDEVICE constexpr size_type size() const noexcept { return (N * M); }

  HOSTDEVICE constexpr size_type max_size() const noexcept { return (N * M); }

  HOSTDEVICE constexpr size_type rows() const noexcept { return N; }

  HOSTDEVICE constexpr size_type cols() const noexcept { return M; }

  // element access:
  HOSTDEVICE reference operator[](size_type n) { return elemns[n]; }

  HOSTDEVICE reference operator()(size_type x, size_type y) { return elemns[x * M + y]; }

  HOSTDEVICE reference at(size_type n) { return elemns[n]; }

  HOSTDEVICE const_reference cat(size_type n) const { return elemns[n]; }

  HOSTDEVICE reference at(size_type x, size_type y) { return elemns[x * M + y]; }

  HOSTDEVICE const_reference cat(size_type x, size_type y) const { return elemns[x * M + y]; }

  HOSTDEVICE reference front() { return elemns[0]; }

  HOSTDEVICE const_reference cfront() const { return elemns[0]; }

  HOSTDEVICE reference back() { return elemns[N - 1]; }

  HOSTDEVICE const_reference cback() const { return elemns[N - 1]; }

  HOSTDEVICE value_type *data() noexcept { return elemns; }

  HOSTDEVICE const value_type *cdata() const noexcept { return elemns; }

  HOSTDEVICE void row_shuffle() {
    typedef thrust::uniform_int_distribution <size_type> distr_t;
    typedef typename distr_t::param_type param_t;

    thrust::minstd_rand g;
    distr_t D;
    for (size_type i = N - 1; i > 0; --i) {
      using thrust::swap;
      auto dd = D(g, param_t(0, i));
      for (size_type j = 0; j < M; j++)
        swap(at(i, j), at(dd, j));
    }
  }

  HOSTDEVICE void column_shuffle() {
    typedef thrust::uniform_int_distribution <size_type> distr_t;
    typedef typename distr_t::param_type param_t;

    thrust::minstd_rand g;
    distr_t D;
    for (size_type i = M - 1; i > 0; --i) {
      using thrust::swap;
      auto dd = D(g, param_t(0, i));
      for (size_type j = 0; j < N; j++)
        swap(at(j, i), at(j, dd));
    }
  }

  // Operators
  HOSTDEVICE selfref operator+=(self x) {
    for (int i = 0; i < size(); i++)
      at(i) += x[i];
  }

  HOSTDEVICE selfref operator-=(self x) {
    for (int i = 0; i < size(); i++)
      at(i) -= x[i];
  }

  __host__ pointer get_device_pointer() {
    pointer d_t = nullptr;
    if (!in_device) {
      gpuErrchk(cudaMalloc((void **) &d_t, N * M * sizeof(value_type)));
      gpuErrchk(cudaMemcpy(d_t, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
      in_device = true;
      d_elemns = d_t;
    }
    return d_t;
  }

  __host__ void refresh_from_device() {
    if (in_device) gpuErrchk(cudaMemcpy(elemns, d_elemns, N * M * sizeof(value_type), cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_to_device() {
    if (in_device) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
  }

  __host__ void release_device_data() {
    if (in_device) gpuErrchk(cudaFree(d_elemns));
  }
};

#endif // CPPNNET_CUMATRIX_CUH