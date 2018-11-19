#ifndef CPPNNET_CUARRAY_CUH
#define CPPNNET_CUARRAY_CUH

#include "misc.cuh"

template<class T, int N>
struct cuarray {
  // types:
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
  HOSTDEVICE cuarray() {
    elemns = new T[N];
  }

  HOSTDEVICE cuarray(pointer &data, bool do_delete = true) {
    elemns = data;
    dodelete = do_delete;
  }

  HOSTDEVICE ~cuarray() {
    if (dodelete) delete[] elemns;
    if (in_device) gpuErrchk(cudaFree(d_elemns));
  }

  // iterators
  HOSTDEVICE iterator begin() noexcept { return iterator(data()); }

  HOSTDEVICE const_iterator cbegin() const noexcept { return const_iterator(data()); }

  HOSTDEVICE iterator end() noexcept { return iterator(data() + N); }

  HOSTDEVICE const_iterator cend() const noexcept { return const_iterator(data() + N); }

  // Capacity
  HOSTDEVICE constexpr size_type size() const noexcept { return N; }

  HOSTDEVICE constexpr size_type max_size() const noexcept { return N; }

  // element access:
  HOSTDEVICE reference operator[](size_type n) { return elemns[n]; }

  HOSTDEVICE reference at(size_type n) { return elemns[n]; }

  HOSTDEVICE const_reference cat(size_type n) const { return elemns[n]; }

  HOSTDEVICE reference front() { return elemns[0]; }

  HOSTDEVICE const_reference cfront() const { return elemns[0]; }

  HOSTDEVICE reference back() { return elemns[N - 1]; }

  HOSTDEVICE const_reference cback() const { return elemns[N - 1]; }

  HOSTDEVICE value_type *data() noexcept { return elemns; }

  HOSTDEVICE const value_type *cdata() const noexcept { return elemns; }

  __host__ pointer get_device_pointer() {
    pointer d_t = nullptr;
    if (!in_device) {
      gpuErrchk(cudaMalloc((void **) &d_t, N * sizeof(value_type)));
      gpuErrchk(cudaMemcpy(d_t, elemns, N * sizeof(value_type), cudaMemcpyHostToDevice));
      in_device = true;
      d_elemns = d_t;
    }
    return d_t;
  }

  __host__ void refresh_from_device() {
    if (in_device) gpuErrchk(cudaMemcpy(elemns, d_elemns, N * sizeof(value_type), cudaMemcpyDeviceToHost));
  }

  __host__ void refresh_to_device() {
    if (in_device) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * sizeof(value_type), cudaMemcpyHostToDevice));
  }

  __host__ void delete_device_data() {
    if (in_device) gpuErrchk(cudaFree(d_elemns));
  }
};

#endif // CPPNNET_CUARRAY_CUH