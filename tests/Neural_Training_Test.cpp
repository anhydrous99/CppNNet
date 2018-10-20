#include "Activation_Functions.h"
#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "Normalizer.h"
#include "CSV_Importer.h"
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Error: Wrong number of arguments. Exiting...\n";
    return 1;
  }

  // Create Layers
  std::shared_ptr<Neural_Layer> layer1 = std::make_shared<Neural_Layer>(5, 1, HyperbolicTan_Function);
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

  // Create Derivative Function Vector
  std::vector<function> derv_funs;
  derv_funs.push_back(HyperbolicTan_Function_D);
  derv_funs.push_back(Identity_Function_D);

  // Create Trainer
  Neural_Trainer trainer(layer2, derv_funs);

  // Get Shuffle Idnices

  // Train
  for (int i = 0, sizei = 10000; i < sizei; i++)
  {
    std::vector<int> idxs = trainer.shuffle_indices(samples.size());
    for (int j = 0, sizej = samples.size(); j < sizej; j++)
      trainer.train_sample(normed_samples[idxs[j]], normed_targets[idxs[j]]);
  }

  // Calculate Error
  float mse = layer2->mse(normed_samples, normed_targets);
  std::cout << "MSE: " << mse << std::endl;
  return (0.1 < mse);
}
