#include "CUDA_Neural_Trainer.h"

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
