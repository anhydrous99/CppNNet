//
// Created by Armando Herrera III
// Uses Stochastic Gradient Descent w/ back propagation to train a Neural Network.
//

#ifndef CPPNNET_NEURAL_TRAINER_H
#define CPPNNET_NEURAL_TRAINER_H

#include "Neural_Layer.h"

class Neural_Trainer {
protected:
  std::vector<std::shared_ptr<Neural_Layer>> _neur_ptrs;
  std::vector<function> _daf;
  float _learning_rate = 0.01;

public:
  Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, std::vector<function> derv_fun);

  Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, std::vector<function> derv_fun,
                 float learning_rate);

  Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun);

  Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun, float learning_rate);

  ~Neural_Trainer() = default;

  virtual void train_sample(const Evector &s, const Evector &t);

  virtual void train_batch(const std::vector<Evector> &s, const std::vector<Evector> &t);

  void train_minibatch(const std::vector<Evector> &s, const std::vector<Evector> &t, unsigned long batch_size);

  std::vector<int> shuffle_indices(int nindices);
};

#endif // CPPNNET_NEURAL_TRAINER_H
