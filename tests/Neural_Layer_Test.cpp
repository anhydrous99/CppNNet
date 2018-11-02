#include "Neural_Layer.h"
#include "Normalizer.h"
#include <iostream>

int main() {
  std::cout << "Testing Neural_Layer\n";
  // Declare Weights and bias matrices and vectors
  Ematrix Weights_Layer1(5, 1);
  Evector Bias_Layer1(5);

  Ematrix Weights_Layer2(1, 5);
  Evector Bias_Layer2(1);

  // Add weights to matrices and vectors
  Weights_Layer1 << 6.0140, -5.1062, 5.0833, 3.8055, -3.4355;
  Bias_Layer1 << -4.9522, 2.4840, -0.8271, 2.3480, -3.5037;
  Weights_Layer2 << 0.8756, 1.3115, 0.3390, -0.7949, -1.9944;
  Bias_Layer2 << -0.9361;

  // Declare Activation Functions for 1st layer
  // From Activation_Functions.h
  // Neural_Layer has linear function built in as default
  activation_function activ_func_1st_layer = activation_function::HyperbolicTan;

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1(new Neural_Layer(Weights_Layer1,
                                                        Bias_Layer1,
                                                        activ_func_1st_layer));
  std::shared_ptr<Neural_Layer> layer2(new Neural_Layer(Weights_Layer2,
                                                        Bias_Layer2,
                                                        layer1));

  // Create Input Vector
  Evector input_vec(1);
  input_vec << 0;

  // Normalize input vector to be between [1,-1]
  Evector xoffset(1);
  xoffset << 0;
  Evector gain(1);
  gain << 0.200475452649894;
  float ymin = -1;

  // Normalize
  Normalizer nor1(input_vec, xoffset, gain, ymin);

  // Feed through network
  Evector output = layer2->feedforward(input_vec);

  // Denormalize
  nor1.reverse(output);

  return 0;
}
