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

  cumatrix<float> Sample_Matrix(ss);
  cumatrix<float> Target_Matrix(tt);
}
