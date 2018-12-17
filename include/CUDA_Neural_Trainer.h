//
// Created by Armando Herrera on 11/18/18.
//

#ifndef CPPNNET_CUDA_NEURAL_TRAINER_H
#define CPPNNET_CUDA_NEURAL_TRAINER_H

#include "Neural_Layer.h"

namespace CppNNet {

  class CUDA_Neural_Trainer {
  protected:
    std::vector<std::shared_ptr<Neural_Layer>> _neur_ptrs;
    std::vector<function> _daf;
    float _learning_rate = 0.01;

  public:
    explicit CUDA_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr);

    CUDA_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, float learning_rate);

    explicit CUDA_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr);

    CUDA_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, float learning_rate);

    ~CUDA_Neural_Trainer() = default;

    void train_minibatch(const std::vector<Evector> &s, const std::vector<Evector> &t, unsigned long batch_size,
                         bool shuffle = true);
  };

}

#endif //CPPNNET_CUDA_NEURAL_TRAINER_H
