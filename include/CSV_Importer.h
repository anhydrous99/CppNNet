#ifndef CSV_IMPORTER_H
#define CSV_IMPORTER_H

#include <Eigen/Core>

#include <string>
#include <vector>

class CSV_Importer
{
private:
  std::string _filename;
  char _delim = ',';
  bool _hasdata = false;
  int _sof, _sot;
  std::vector<std::string> _data;

  void ObtainData();
public:
  CSV_Importer(std::string filename, int size_of_samples, int size_of_targets);
  CSV_Importer(std::string filename, int size_of_samples, int size_of_targets, char delimiter);

  std::vector<std::string> GetData();
  std::vector<Eigen::VectorXf> GetSamples();
  std::vector<Eigen::VectorXf> GetTargets();
};

#endif // CSV_IMPORTER_H
