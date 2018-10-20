#include <CSV_Importer.h>
#include <Normalizer.h>

#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Error: Wrong number of arguments. Exiting...\n";
    return 1;
  }

  // Import Data
  std::string path = argv[1];
  CSV_Importer imp(path, 1, 1);
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
