#include "Neural_Layer.h"
#include "MSGD_Neural_Trainer.h"
#include "Normalizer.h"
#include "CSV_Importer.h"
#include <iostream>

using namespace CppNNet;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: Wrong number of arguments. Exiting...\n";
    return 1;
  }

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(5, 1, activation_function::HyperbolicTan);
  std::shared_ptr<Neural_Layer> layer2 = std::make_shared<Neural_Layer>(1, 5, layer1);

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, 1, 1);
  std::vector<Evector> samples = imp.GetSamples();
  std::vector<Evector> targets = imp.GetTargets();

  // Normalize Data
  Normalizer samplen(samples, -1, 1);
  Normalizer targetn(targets, -1, 1);
  std::vector<Evector> normed_samples = samplen.get_batch_norm(samples);
  std::vector<Evector> normed_targets = targetn.get_batch_norm(targets);

  // Create Trainer
  Neural_Trainer trainer(layer2);

  // Train
  for (int i = 0, sizei = 2000; i < sizei; i++) {
    std::vector<int> idxs = trainer.shuffle_indices(samples.size());
    for (int j = 0, sizej = samples.size(); j < sizej; j++)
      trainer.train_sample(normed_samples[idxs[j]], normed_targets[idxs[j]]);
  }

  // Calculate Error
  float mse = layer2->mse(normed_samples, normed_targets);
  float rmse = layer2->rmse(normed_samples, normed_targets);
  float mae = layer2->mae(normed_samples, normed_targets);
  float mpe = layer2->mpe(normed_samples, normed_targets);
  float mape = layer2->mape(normed_samples, normed_targets);
  float r2 = layer2->r2_avg(normed_samples, normed_targets);
  float aic = layer2->aic(normed_samples, normed_targets);
  float aicc = layer2->aicc(normed_samples, normed_targets);
  float bic = layer2->bic(normed_samples, normed_targets);
  std::cout << "MSE:  " << mse << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;
  std::cout << "MAE:  " << mae << std::endl;
  std::cout << "MPE:  " << mpe << std::endl;
  std::cout << "MAPE: " << mape << std::endl;
  std::cout << "R^2:  " << r2 << std::endl;
  std::cout << "AIC:  " << aic << std::endl;
  std::cout << "AICC: " << aicc << std::endl;
  std::cout << "BIC   " << bic << std::endl;
  return (0.1 < mse);
}
