#ifndef NEURAL_TRAINER_H
#define NEURAL_TRAINER_H

#include "Neural_Layer.h"
#include <vector>

class Neural_Trainer
{
private:
  std::vector<std::shared_ptr<Neural_Layer>> _neur_ptrs;
  std::vector<function> _daf;
  float _learning_rate = 0.01;

public:
  Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, std::vector<function> derv_fun);
  Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr, std::vector<function> derv_fun, float learning_rate);

  Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun);
  Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun, float learning_rate);

  ~Neural_Trainer() {}

  void train_sample(Evector s, Evector t);

  std::vector<int> shuffle_indices(int nindices);
};

#endif // NEURAL_TRAINER_H
