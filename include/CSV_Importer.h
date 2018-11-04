//
// Created by Armando Herrera III
// Imports or Downloads data in the CSV format
//

#ifndef CPPNNET_CSV_IMPORTER_H
#define CPPNNET_CSV_IMPORTER_H

#include <Eigen/Core>
#include <string>
#include <vector>

namespace CppNNet {

  typedef struct MemoryStruct MemoryStruct;
  struct MemoryStruct {
    unsigned char *memory;
    size_t size;
  };

  class CSV_Importer {
  private:
    std::string _filename;
    char _delim = ',';
    bool _hasdata = false;
    bool _curlinit = false;
    bool _reverse = false;
    int _sof, _sot;
    int _start_idx = 0;
    std::vector<std::string> _data;

    void ObtainData();

    struct MemoryStruct Binary_Downloader();

    int doinflate(const MemoryStruct *src, MemoryStruct *dst);

    void parse(std::string &to_parse);

    void zerr(int ret);

  public:
    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, bool reverse);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter, bool reverse);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx, bool reverse);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx, char delimiter);

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx, char delimiter,
                 bool reverse);

    ~CSV_Importer();

    std::vector<std::string> GetData();

    std::vector<Eigen::VectorXf> GetSamples();

    std::vector<Eigen::VectorXf> GetTargets();
  };

}

#endif // CPPNNET_CSV_IMPORTER_H
