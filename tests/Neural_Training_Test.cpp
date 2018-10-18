#include "Activation_Functions.h"
#include "Neural_Layer.h"
#include "Neural_Trainer.h"
#include "Normalizer.h"
#include "CSV_Importer.h"

#include <iostream>
#include <cmath>

int main(int argc, char* argv[])
{
  // Create Layers
  std::shared_ptr<Neural_Layer> layer1(new Neural_Layer(5, 1, HyperbolicTan_Function));
  std::shared_ptr<Neural_Layer> layer2(new Neural_Layer(1, 5, layer1));

  // Import Data
  CSV_Importer imp(argv[1], 1, 1);
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

  Evector input = normed_samples[0];
  Evector unnormed_output = layer2->feedforward(input);
  Evector output = targetn.GetReverse(unnormed_output);

  float e = abs(targets[0][0] - output[0]);

  std::cout << "Ouput Vector:\n" << output << "\nTarget Vector:\n" << targets[0] << std::endl;
  return (0.5 < e) ? 1 : 0 ;
}
