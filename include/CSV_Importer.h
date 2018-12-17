//
// Created by Armando Herrera III
// Imports or Downloads data in the CSV format
//

#ifndef CPPNNET_CSV_IMPORTER_H
#define CPPNNET_CSV_IMPORTER_H

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <string>
#include <vector>

namespace CppNNet {

  typedef struct MemoryStruct MemoryStruct;
  struct MemoryStruct {
    unsigned char *memory;
    size_t size;
  };

  struct DataSet {
    std::vector<Eigen::VectorXf> sample_training_set;
    std::vector<Eigen::VectorXf> sample_validation_set;
    std::vector<Eigen::VectorXf> target_training_set;
    std::vector<Eigen::VectorXf> target_validation_set;
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
    int _end_idx = std::numeric_limits<int>::max();
    unsigned long _val = 10;
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

    CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx, int end_idx);

    ~CSV_Importer();

    std::vector<std::string> GetData();

    std::vector<Eigen::VectorXf> GetSamples();

    std::vector<Eigen::VectorXf> GetTargets();

    unsigned long &GetSetValidationPercentage() { return _val; }

    DataSet GetDataSet();
  };

}

#endif // CPPNNET_CSV_IMPORTER_H
