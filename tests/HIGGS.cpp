#include "Activation_Functions.h"
#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "CSV_Importer.h"
#include "Normalizer.h"
#include <iostream>
#include <chrono>

#include <fenv.h>


int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: wrong number of arguments. Exiting...\n";
    return 1;
  }

  //feenableexcept(FE_INVALID | FE_OVERFLOW);

  // Number of inputs
  int inp = 28;
  // Number of outputs
  int out = 2;
  // Number of neurons
  int nn = 400;

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(nn, inp, Logistic_Function);
  std::shared_ptr<Neural_Layer> layer2 = std::make_shared<Neural_Layer>(out, nn, layer1);

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, inp, out);
  std::vector<Evector> samples = imp.GetSamples();
  std::vector<Evector> targets = imp.GetTargets();

  // Normalize data
  Normalizer samplen(samples, 0, 1);
  std::vector<Evector> normed_samples = samplen.get_batch_norm(samples);

  // create Derivative Function Vectoor
  std::vector<function> derv_funs;
  derv_funs.push_back(Logistic_Function_D);
  derv_funs.push_back(Identity_Function_D);

  // Create Trainer
  Neural_Trainer trainer(layer2, derv_funs);

  // Train
  std::cout << "Starting to Train\n";
  for (int i = 0, sizei = 100; i < sizei; i++) {
    trainer.train_minibatch(normed_samples, targets, 100000);
    std::cout << "Epoch: " << i << std::endl;
  }

  // Calculate Error
  float mse = layer2->mse(normed_samples, targets);

  std::cout << "MSE:  " << mse << std::endl;

  return (0.5 < mse);
}
