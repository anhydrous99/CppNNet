#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "CSV_Importer.h"
#include "Normalizer.h"
#include <iostream>
#include <chrono>

// Stops training in the case of float-point overflow, underflow, or invalid.
#ifdef __linux__
#include <fenv.h>

#else
#pragma float_control( except, on )
#endif

using namespace CppNNet;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: wrong number of arguments. Exiting...\n";
    return 1;
  }

#ifdef __linux__
  feenableexcept(FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
#endif

  // Number of inputs
  int inp = 28;
  // Number of outputs
  int out = 2;

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(150, inp, activation_function::Logistic);
  std::shared_ptr<Neural_Layer> layer2 = std::make_shared<Neural_Layer>(100, 150, layer1,
                                                                        activation_function::Logistic);
  std::shared_ptr<Neural_Layer> layer3 = std::make_shared<Neural_Layer>(out, 100, layer2);

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, inp, out);
  std::vector<Evector> samples = imp.GetSamples();
  std::vector<Evector> targets = imp.GetTargets();

  // Normalize data
  Normalizer samplen(samples, 0, 1);
  std::vector<Evector> normed_samples = samplen.get_batch_norm(samples);

  // Create Trainer
  Neural_Trainer trainer(layer3, 0.0001);

  // Train
  std::cout << "Starting to Train\n";
  for (int i = 0, sizei = 10; i < sizei; i++) {
    trainer.train_minibatch(normed_samples, targets, 100000);
    std::cout << "Epoch: " << i << std::endl;
  }

  // Calculate Error
  float mse = layer3->mse(normed_samples, targets);

  std::cout << "MSE:  " << mse << std::endl;

  return (0.5 < mse);
}
