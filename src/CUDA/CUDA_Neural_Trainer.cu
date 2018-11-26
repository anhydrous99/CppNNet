#include "CUDA_Neural_Trainer.h"
#include "misc.cuh"
#include "util.h"
#include "cuNetwork.cuh"
#include "cumatrix.cuh"

CppNNet::CUDA_Neural_Trainer::CUDA_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs) {
  _neur_ptrs = neural_ptrs;
  unsigned long M = neural_ptrs.size();
  for (unsigned long m = 0; m < M; m++)
    _daf.push_back(neural_ptrs[m]->Get_Derivative_Function());
}

CppNNet::CUDA_Neural_Trainer::CUDA_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs,
                                                  float learning_rate) :
    CUDA_Neural_Trainer(std::move(neural_ptrs)) {
  _learning_rate = learning_rate;
}

CppNNet::CUDA_Neural_Trainer::CUDA_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr) {
  _neur_ptrs = end_neural_ptr->GetVecPtrs();
  unsigned long M = _neur_ptrs.size();
  for (unsigned long m = 0; m < M; m++)
    _daf.push_back(_neur_ptrs[m]->Get_Derivative_Function());
}

CppNNet::CUDA_Neural_Trainer::CUDA_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, float learning_rate) :
    CUDA_Neural_Trainer(std::move(end_neural_ptr)) {
  _learning_rate = learning_rate;
}

void throw_error(std::string error_string) {
  throw new std::runtime_error(error_string + "\n");
}

void CppNNet::CUDA_Neural_Trainer::train_minibatch(const std::vector<CppNNet::Evector> &s,
                                                   const std::vector<CppNNet::Evector> &t, unsigned long batch_size,
                                                   bool shuffle) {
  auto n_iterations = (int) std::floor((double) s.size() / (double) batch_size);

  // Push Weights and Biases to cuNetwork object
  std::vector<Ematrix> Weights_vec;
  std::vector<Evector> Biases_vec;
  for (int m = 0; m < _neur_ptrs.size(); m++) {
    Weights_vec.push_back(_neur_ptrs[m]->_w);
    Biases_vec.push_back(_neur_ptrs[m]->_b);
  }
  cuNetwork<float> network(Weights_vec, Biases_vec);

  // Copy samples to shuffle
  std::vector<Evector> ss = s;
  std::vector<Evector> tt = t;

  // Shuffle samples
  if (shuffle)
    double_shuffle(ss.begin(), ss.end(), tt.begin());

  // Put samples in cumatrix object
  cumatrix<float> Sample_Matrix(ss);
  cumatrix<float> Target_Matrix(tt);

  // Get Device Pointers
  float *d_weights, *d_biases;
  network.get_device_pointers(d_weights, d_biases);
  float *d_samples = Sample_Matrix.get_device_pointer();
  float *d_targets = Target_Matrix.get_device_pointer();
  int *d_dims = network.get_dim_array_dev_ptr();

  // Create Output Sensitivity objects
  cuNetwork<float> sensi(network.N_layers(), network.Get_Dims_ptr());
  float *d_senTa, *d_sen;
  sensi.get_device_pointers(d_senTa, d_sen, false);

  kernel_parameters params = {
      network.N_layers(),
      n_iterations,
      Sample_Matrix.rows()
  };

  BackProp << < 1, 1 >> > (d_weights, d_biases, d_samples, d_targets, d_senTa, d_sen, d_dims, params);

  // Get Calculated Sensitivity
  sensi.refresh_weights_from_device();
  sensi.refresh_biases_from_device();

  // Calculate New Weights
  /// TODO

  // Release device memory
  sensi.release_device_data();
  network.release_device_data();
  Sample_Matrix.release_device_data();
  Target_Matrix.release_device_data();
};
