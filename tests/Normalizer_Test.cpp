#include <CSV_Importer.h>
#include <Normalizer.h>

#include <iostream>

int main(int argc, char* argv[])
{
  // Import Data
  CSV_Importer imp(argv[1], 1, 1);
  std::vector<Eigen::VectorXf> samples = imp.GetSamples();
  std::vector<Eigen::VectorXf> targets = imp.GetTargets();

  // Create Normalizer
  Normalizer norm1(samples, -1, 1);
  Normalizer norm2(targets, -1, 1);

  NormSettings settings1 = norm1.GetSettings();
  NormSettings settings2 = norm2.GetSettings();

  std::cout << "1:offset:\n" << settings1.xoffset << ":gain:\n" << settings1.gain << ":ymin:" << settings1.ymin << std::endl;
  std::cout << "2:offset:\n" << settings2.xoffset << ":gain:\n" << settings2.gain << ":ymin:" << settings2.ymin << std::endl;
  return 0;
}
