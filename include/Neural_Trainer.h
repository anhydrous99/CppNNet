//
// Created by Armando Herrera III
// Uses Stochastic Gradient Descent w/ back propagation to train a Neural Network.
//

#ifndef CPPNNET_NEURAL_TRAINER_H
#define CPPNNET_NEURAL_TRAINER_H

#include "Neural_Layer.h"

namespace CppNNet {

  class Neural_Trainer {
  protected:
    std::vector<std::shared_ptr<Neural_Layer>> _neur_ptrs;
    std::vector<function> _daf;
    float _learning_rate = 0.01;

  public:
    explicit Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr);

    Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, float learning_rate);

    explicit Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr);

    Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, float learning_rate);

    ~Neural_Trainer() = default;

    virtual void train_sample(const Evector &s, const Evector &t);

    virtual void train_batch(const std::vector<Evector> &s, const std::vector<Evector> &t, bool shuffle);

    void train_batch(const std::vector<Evector> &s, const std::vector<Evector> &t);

    void train_minibatch(const std::vector<Evector> &s, const std::vector<Evector> &t, unsigned long batch_size,
                         bool shuffle = true);

    std::vector<unsigned long> shuffle_indices(unsigned long nindices);
  };

}

#endif // CPPNNET_NEURAL_TRAINER_H
