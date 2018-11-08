#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "CSV_Importer.h"
#include "Net_Importer.h"
#include <iostream>
#include <chrono>

using namespace CppNNet;

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
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(10, inp, activation_function::Logistic);
  std::shared_ptr<Neural_Layer> layer2 = std::make_shared<Neural_Layer>(out, 10, layer1);

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, inp, out);
  DataSet dataset = imp.GetDataSet();

  // Create Trainer
  Neural_Trainer trainer(layer2);

  // Train
  auto start1 = std::chrono::steady_clock::now();
  for (int i = 0, sizei = 2000; i < sizei; i++) {
    trainer.train_batch(dataset.sample_training_set, dataset.target_training_set);
  }
  auto end1 = std::chrono::steady_clock::now();

  // Calculate error
  float mse = layer2->mse(dataset.sample_validation_set, dataset.target_validation_set);
  float rmse = layer2->rmse(dataset.sample_validation_set, dataset.target_validation_set);
  float mae = layer2->mae(dataset.sample_validation_set, dataset.target_validation_set);
  float mpe = layer2->mpe(dataset.sample_validation_set,
                          dataset.target_validation_set);   // mpe and mape return -nan and inf
  float mape = layer2->mape(dataset.sample_validation_set,
                            dataset.target_validation_set); //  since targets contain zeros
  float r2 = layer2->r2_avg(dataset.sample_validation_set, dataset.target_validation_set);
  float aic = layer2->aic(dataset.sample_training_set, dataset.target_training_set);
  float aicc = layer2->aicc(dataset.sample_training_set, dataset.target_training_set);
  float bic = layer2->bic(dataset.sample_training_set, dataset.target_training_set);
  auto end2 = std::chrono::steady_clock::now();

  std::cout << "MSE:            " << mse << std::endl;
  std::cout << "RMSE:           " << rmse << std::endl;
  std::cout << "MAE:            " << mae << std::endl;
  std::cout << "MPE:            " << mpe << std::endl;
  std::cout << "MAPE:           " << mape << std::endl;
  std::cout << "R^2:            " << r2 << std::endl;
  std::cout << "AIC:            " << aic << std::endl;
  std::cout << "AICC:           " << aicc << std::endl;
  std::cout << "BIC             " << bic << std::endl;
  std::cout << "It took         " << std::chrono::duration_cast<std::chrono::seconds>(end1 - start1).count()
            << " seconds"
            << std::endl;
  std::cout << "Statistics took " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count()
            << " milliseconds" << std::endl;

  // Export network
  Net_Importer exprt("iris_network.json");
  exprt.writeNet(layer2);

  // Import network
  std::shared_ptr<Neural_Layer> network2 = exprt.readNet_endptr();
  mse = network2->mse(dataset.sample_validation_set, dataset.target_validation_set);
  std::cout << "imported network MSE: " << mse << std::endl;

  return (0.5 < mse);
}
