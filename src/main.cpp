#include "Neural_Trainer.h"
#include "Neural_Layer.h"
#include "Activation_Functions.h"
#include "Normalizer.h"

#include "CSV_Importer.h"

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
  function activation_function = HyperbolicTan_Function;

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

  std::cout << "Testing CSV Importer\n";

  // Create CSV_Importer and get samples and targets to train
  CSV_Importer imp("../training_data/simpledata_set.csv", 1, 1);
  std::vector<Eigen::VectorXf> samples = imp.GetSamples();
  std::vector<Eigen::VectorXf> targets = imp.GetTargets();

  // Print both sample and target vectors to make sure they loaded correctly
  std::cout << "Printing sample vectors\n";
  for (int i = 0, size = samples.size(); i < size; i++)
    std::cout << samples[i] << std::endl;

  std::cout << "Printing target vectors\n";
  for (int i = 0, size = targets.size(); i < size; i++)
    std::cout << targets[i] << std::endl;

  return 0;
}
