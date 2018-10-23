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
  auto start1 = std::chrono::steady_clock::now();
  for (int i = 0, sizei = 2000; i < sizei; i++) {
    trainer.train_batch(samples, targets);
  }
  auto end1 = std::chrono::steady_clock::now();

  // Calculate error
  float mse = layer2->mse(samples, targets);
  float rmse = layer2->rmse(samples, targets);
  float mae = layer2->mae(samples, targets);
  float mpe = layer2->mpe(samples, targets);   // mpe and mape return -nan and inf
  float mape = layer2->mape(samples, targets); //  since targets contain zeros
  float r2 = layer2->r2(samples, targets);
  auto end2 = std::chrono::steady_clock::now();

  std::cout << "MSE:            " << mse << std::endl;
  std::cout << "RMSE:           " << rmse << std::endl;
  std::cout << "MAE:            " << mae << std::endl;
  std::cout << "MPE:            " << mpe << std::endl;
  std::cout << "MAPE:           " << mape << std::endl;
  std::cout << "R^2:            " << r2 << std::endl;
  std::cout << "It took         " << std::chrono::duration_cast<std::chrono::seconds>(end1 - start1).count()
            << " seconds"
            << std::endl;
  std::cout << "Statistics took " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count()
            << " milliseconds";
  return (0.5 < mse);
}
