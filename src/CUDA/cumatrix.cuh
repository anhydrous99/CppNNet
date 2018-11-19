#ifndef CPPNNET_CUMATRIX_CUH
#define CPPNNET_CUMATRIX_CUH

#include "misc.cuh"

template<class T, int N, int M>
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

  // Constructor & Destructor
  HOSTDEVICE cumatrix() {
    elemns = new T[N * M];
  }

  HOSTDEVICE cumatrix(pointer data, bool do_delete = true) {
    elemns = data;
    dodelete = do_delete;
  }

  HOSTDEVICE ~cumatrix() {
    if (dodelete) {
      delete[] elemns;
      if (in_device) gpuErrchk(cudaFree(d_elemns));
    }
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
    pointer d_t;
    gpuErrchk(cudaMalloc((void **) &d_t, N * M * sizeof(value_type)));
    gpuErrchk(cudaMemcpy(d_t, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
    in_device = true;
    d_elemns = d_t;
    return d_t;
  }

  __host__ void refresh_from_device() {
    if (in_device) gpuErrchk(cudaMemcpy(elemns, d_elemns, N * M * sizeof(value_type), cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_to_device() {
    if (in_device) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
  }

  __host__ void delete_device_data() {
    if (in_device) gpuErrchk(cudaFree(d_elemns));
  }
};

// Element Wise Operators
template<class T, int N, int M>
HOSTDEVICE cumatrix<T, N, M> operator+(cumatrix<T, N, M> a, cumatrix<T, N, M> b) {
  cumatrix<T, N, M> output;
  for (int i = 0, s = a.size(); i < s; i++)
    output[i] = a[i] + b[i];
  return output;
}

template<class T, int N, int M>
HOSTDEVICE cumatrix<T, N, M> operator-(cumatrix<T, N, M> a, cumatrix<T, N, M> b) {
  cumatrix<T, N, M> output;
  for (int i = 0, s = a.size(); i < s; i++)
    output[i] = a[i] - b[i];
  return output;
}

#endif // CPPNNET_CUMATRIX_CUH