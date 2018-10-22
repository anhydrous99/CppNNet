#include "Activation_Functions.h"
#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "CSV_Importer.h"
#include <iostream>
#include <chrono>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Error: wrong number of arguments. Exiting...\n";
    return 1;
  }

  // Number of inputs
  int inp = 4;
  // Number of outputs
  int out = 3;

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(10, inp, Logistic_Function);
  std::shared_ptr<Neural_Layer> layer2 = std::make_shared<Neural_Layer>(out, 10, layer1);

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, inp, out);
  std::vector<Evector> samples = imp.GetSamples();
  std::vector<Evector> targets = imp.GetTargets();

  // Create Derivative Function Vector
  std::vector<function> derv_funs;
  derv_funs.push_back(Logistic_Function_D);
  derv_funs.push_back(Identity_Function_D);

  // Create Trainer
  Neural_Trainer trainer(layer2, derv_funs);

  // Train
  auto start = std::chrono::steady_clock::now();
  for (int i = 0, sizei = 5000; i < sizei; i++) {
    trainer.train_batch(samples, targets);
  }
  auto end = std::chrono::steady_clock::now();

  // Calculate error
  float mse = layer2->mse(samples, targets);
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "It took " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds"
            << std::endl;
  return (0.1 < mse);
}
