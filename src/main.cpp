#include "Neural_Layer.h"

#include <iostream>
#include <cmath>

int main(void)
{
  Ematrix Weights_Layer1(5, 1);
  Evector Bias_Layer1(5);

  Ematrix Weights_Layer2(1, 5);
  Evector Bias_Layer2(1);

  Weights_Layer1 << 6.0140, -5.1062, 5.0833, 3.8055, -3.4355;
  Bias_Layer1    << -4.9522, 2.4840, -0.8271, 2.3480, -3.5037;
  Weights_Layer2 << 0.8756, 1.3115, 0.3390, -0.7949, -1.9944;
  Bias_Layer2    << -0.9361;
  function activation_function = [](float x){ return 2/(1+exp(-2*x)) - 1; };

  std::shared_ptr<Neural_Layer> layer1(new Neural_Layer(Weights_Layer1, Bias_Layer1, activation_function));
  std::shared_ptr<Neural_Layer> layer2(new Neural_Layer(Weights_Layer2, Bias_Layer2, layer1));

  Evector input_vec(1);
  input_vec << 0;

  // Normalize input vector to be between [1,-1]
  input_vec[0] = input_vec[0] * 0.200475452649894 - 1;

  Evector output = layer2->feedforward(input_vec);

  // Undo Neormalization
  output[0] = (output[0] + 1) / 0.2;

  std::cout << output << std::endl;

  return 0;
}
