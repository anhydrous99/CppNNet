//
// Created by Armando Herrera III on 01/11/18.
// Imports and Exports Neural Networks in JSON format
//

#ifndef CPPNNET_NET_IMPORTER_H
#define CPPNNET_NET_IMPORTER_H

#include "Neural_Layer.h"
#include <string>

class Net_Importer {
private:
  std::string _filename;

  std::string readfile();

  void writefile(std::string content);

public:

  explicit Net_Importer(std::string filename);

  explicit Net_Importer(std::string &filename);

  ~Net_Importer() = default;

  std::shared_ptr<Neural_Layer> readNet_endptr();

  std::vector<std::shared_ptr<Neural_Layer>> readNet_vecptr();

  void writeNet(std::shared_ptr<Neural_Layer> ptr);

  void writeNet(std::vector<std::shared_ptr<Neural_Layer>> ptrs);
};

#endif // CPPNNET_NET_IMPORTER_H
