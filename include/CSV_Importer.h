#ifndef CSV_IMPORTER_H
#define CSV_IMPORTER_H

#include <Eigen/Core>
#include <string>
#include <vector>

class CSV_Importer {
private:
  std::string _filename;
  char _delim = ',';
  bool _hasdata = false;
  bool _curlinit = false;
  int _sof, _sot;
  int _start_idx = 0;
  std::vector<std::string> _data;

  void ObtainData();

  std::string Downloader();

  void parse(std::string &to_parse);

public:
  CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets);

  CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter);

  CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx);

  CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx, char delimiter);

  ~CSV_Importer();

  std::vector<std::string> GetData();

  std::vector<Eigen::VectorXf> GetSamples();

  std::vector<Eigen::VectorXf> GetTargets();
};

#endif // CSV_IMPORTER_H
