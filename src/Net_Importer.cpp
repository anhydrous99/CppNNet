//
// Created by Armando Herrera III on 11/01/18.
//

#include "Net_Importer.h"

#include <iostream>
#include <fstream>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

Net_Importer::Net_Importer(std::string &filename) : _filename(filename) {
}

std::string Net_Importer::readfile() {
  std::string line;
  std::string full;
  std::ifstream file(_filename);
  if (file.is_open()) {
    while (std::getline(file, line))
      full += line;
    file.close();
  }
  return full;
}

void Net_Importer::writefile(std::string content) {
  std::ofstream file(_filename);
  if (file.is_open()) {
    file << content;
    file.close();
  }
}

std::vector<std::shared_ptr<Neural_Layer>> Net_Importer::readNet_vecptr() {
  std::string net_str = readfile();
  std::vector<std::shared_ptr<Neural_Layer>> output;
  rapidjson::Document d;
  d.Parse(net_str.c_str());
  /* Get number of Layers */
  assert(d.IsObject());
  assert(d["nlayers"].IsInt());
  long M = d["nlayers"].GetInt();
  assert(d["layers"].IsArray());
  rapidjson::Value &layers = d["layers"];
  for (long m = 0; m < M; m++) {
    rapidjson::Value &layer = layers[m];

    assert(layer["nneurons"].IsInt());
    assert(layer["ninputs"].IsInt());
    long nneurons = layer["nneurons"].GetInt();
    long ninputs = layer["ninputs"].GetInt();

    assert(layer["activ_function"].IsInt());
    auto func = (activation_function) layer["activ_function"].GetInt();

    Ematrix w(nneurons, ninputs);
    Evector b(nneurons);
    rapidjson::Value &weights = layer["weights"];
    rapidjson::Value &biases = layer["biases"];
    assert(weights.IsArray());
    assert(biases.IsArray());
    for (long nn = 0; nn < nneurons; nn++) {
      rapidjson::Value &weightsrow = layer[nn];
      assert(weightsrow.IsArray());
      for (long ni = 0; ni < ninputs; ni++) {
        assert(weights[ni].IsFloat());
        w(nn, ni) = weights[ni].GetFloat();
      }

      assert(biases[nn].IsFloat());
      b[nn] = biases[nn].GetFloat();
    }

    if (m == 0) {
      output.emplace_back(std::make_shared<Neural_Layer>(w, b, func));
    } else {
      output.emplace_back(std::make_shared<Neural_Layer>(w, b, output[m - 1], func));
    }
  }

  return output;
}

std::shared_ptr<Neural_Layer> Net_Importer::readNet_endptr() {
  std::vector<std::shared_ptr<Neural_Layer>> ptrs = readNet_vecptr();
  return ptrs[ptrs.size() - 1];
}

void Net_Importer::writeNet(std::vector<std::shared_ptr<Neural_Layer>> ptrs) {

  rapidjson::Document d;
  d.SetObject();
  rapidjson::Document::AllocatorType &allocator = d.GetAllocator();

  /* Add Number of Layers */
  unsigned long M = ptrs.size();
  d.AddMember("nlayers", M, allocator);

  /* Network Layers */
  rapidjson::Value layers(rapidjson::kArrayType);
  for (unsigned m = 0; m < M; m++) {
    rapidjson::Value layer;
    long nneurons = ptrs[m]->GetNNeurons();
    long ninputs = ptrs[m]->GetNInputs();
    /* Add number of neurons in layer */
    layer.AddMember("nneurons", nneurons, allocator);

    /* Add number of inputs in layer */
    layer.AddMember("ninputs", ninputs, allocator);

    /* Add the activation function */
    layer.AddMember("activ_function", (int) ptrs[m]->Current_Activation_Function(), allocator);

    /* Weights in Layer */
    Ematrix w = ptrs[m]->GetWeights();
    rapidjson::Value weights(rapidjson::kArrayType);
    for (long nn = 0; nn < nneurons; nn++) {
      rapidjson::Value weightsrow(rapidjson::kArrayType);
      for (long ni = 0; ni < ninputs; ni++) {
        weightsrow.PushBack(w(nn, ni), allocator);
      }
      weights.PushBack(weightsrow, allocator);
    }
    layer.AddMember("weights", weights, allocator);

    /* biases in layer */
    Evector b = ptrs[m]->GetBiases();
    rapidjson::Value biases(rapidjson::kArrayType);
    for (long nn = 0; nn < nneurons; nn++) {
      biases.PushBack(b[nn], allocator);
    }
    layer.AddMember("biases", biases, allocator);
    layers.PushBack(layer, allocator);

    /* add activation function */
  }
  d.AddMember("layers", layers, allocator);

  /* create string with rapidjson */
  rapidjson::StringBuffer strbuf;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
  d.Accept(writer);

  writefile(strbuf.GetString());
}

void Net_Importer::writeNet(std::shared_ptr<Neural_Layer> ptr) {
  std::vector<std::shared_ptr<Neural_Layer>> ptrs = ptr->GetVecPtrs();
  writeNet(ptrs);
}