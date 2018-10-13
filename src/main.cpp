#include "Neural_Trainer.h"
#include "Neural_Layer.h"
#include "Normalizer.h"

#include <iostream>
#include <cmath>

int main(void)
{
  std::cout << "Testing Neural_Layer\n";
  // Declare Weights and bias matrices and vectors
  Ematrix Weights_Layer1(5, 1);
  Evector Bias_Layer1(5);

  Ematrix Weights_Layer2(1, 5);
  Evector Bias_Layer2(1);

  // Add weights to matries and vectors
  Weights_Layer1 << 6.0140, -5.1062, 5.0833, 3.8055, -3.4355;
  Bias_Layer1    << -4.9522, 2.4840, -0.8271, 2.3480, -3.5037;
  Weights_Layer2 << 0.8756, 1.3115, 0.3390, -0.7949, -1.9944;
  Bias_Layer2    << -0.9361;

  std::cout << "2-Layer Neural_Network\nw1: " << Weights_Layer1 <<
    "\nb1: " << Bias_Layer1 << "\nw2: " << Weights_Layer2 << "\nb2:" << 
    Bias_Layer2 << std::endl;

  // Declare Activation Function for 1st layer
  function activation_function = [](float x){ return 2/(1+exp(-2*x)) - 1; };

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1(new Neural_Layer(Weights_Layer1, Bias_Layer1, activation_function));
  std::shared_ptr<Neural_Layer> layer2(new Neural_Layer(Weights_Layer2, Bias_Layer2, layer1));

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

  // Feed throught network
  Evector output = layer2->feedforward(input_vec);

  // Denormalize
  nor1.reverse(output);

  // Print network output
  std::cout << output << std::endl;

  std::cout << "Testing Neural-Trainer - single sample\n";

  // Create new network with randomly initialized weights
  std::shared_ptr<Neural_Layer> layer11(new Neural_Layer(5,1));
  std::shared_ptr<Neural_Layer> layer12(new Neural_Layer(1,5, layer11));

  // Put pointers in vector
  std::vector<std::shared_ptr<Neural_Layer>> neural_pointers;
  neural_pointers.push_back(layer11);
  neural_pointers.push_back(layer12);

  // Put derivative of the activation functions in a vector
  std::vector<function> derv_funcs;
  derv_funcs.push_back([](float x){ return 1; });
  derv_funcs.push_back([](float x){ return 1; });

  // Create and initialize sample vector and target vector
  Evector sample(1);
  Evector target(1);

  sample << 0.5;
  target << 0.7;

  std::cout << "Starting Layer 1 Weights \n" << 
    layer11->GetWeights() << "\nLayer 2 Weights\n" << layer12->GetWeights() << std::endl;

  // Create trainer
  Neural_Trainer trainer(neural_pointers, derv_funcs, 0.1);

  // Train with single sample
  trainer.train_sample(sample, target);

  std::cout << "After Single Training Layer 1 Weights \n" << 
    layer11->GetWeights() << "\nLayer 2 Weights\n" << layer12->GetWeights() << std::endl;

  return 0;
}
