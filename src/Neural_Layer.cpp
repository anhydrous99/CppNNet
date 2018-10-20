#include "Neural_Layer.h"

#include <random>

Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer, function activation_function) :
    Neural_Layer(Weights, Bias, previous_layer) {
  _activ_func = activation_function;
}

Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer) :
    Neural_Layer(Weights, Bias) {
  _prev_layer = previous_layer;
}

Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, function activation_function) :
    Neural_Layer(Weights, Bias) {
  _activ_func = activation_function;
}

Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias) {
  _w = Weights;
  _b = Bias;
}

Neural_Layer::Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer, function activation_function) :
    Neural_Layer(nneurons, ninputs, previous_layer) {
  _activ_func = activation_function;
}

Neural_Layer::Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer) :
    Neural_Layer(nneurons, ninputs) {
  _prev_layer = previous_layer;
}

Neural_Layer::Neural_Layer(int nneurons, int ninputs, function activation_function) :
    Neural_Layer(nneurons, ninputs) {
  _activ_func = activation_function;
}

Neural_Layer::Neural_Layer(int nneurons, int ninputs) : _w(nneurons, ninputs), _b(nneurons) {
  // Obtain seed
  std::random_device rd;
  // Standard merseen_twister_engine
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  // Fill the weights with random numbers
  for (size_t i = 0, size = _w.size(); i < size; i++)
    *(_w.data() + i) = dis(gen);

  // Fill the bias with random numbers
  for (size_t i = 0, size = _b.size(); i < size; i++)
    *(_b.data() + i) = dis(gen);
}

Evector Neural_Layer::feedforward(Evector input) {
  Evector a = (_prev_layer) ? _prev_layer->feedforward(input) : input;
  Evector n = _w * a + _b;

  for (size_t i = 0, size = n.size(); i < size; i++)
    n[i] = _activ_func(n[i]);

  return n;
}

std::vector<Evector> Neural_Layer::feedforward_batch(std::vector<Evector> input) {
  std::vector<Evector> output;
  for (auto &item: input)
    output.push_back(feedforward(item));
  return output;
}

std::vector<std::shared_ptr<Neural_Layer>> Neural_Layer::GetVecPtrs() {
  std::vector<std::shared_ptr<Neural_Layer>> output;

  if (_prev_layer)
    output = _prev_layer->GetVecPtrs();
  output.push_back(shared_from_this());
  return output;
}

float Neural_Layer::mse(std::vector<Evector> input, std::vector<Evector> target) {
  std::vector<Evector> output = feedforward_batch(input);

  float mse = 0.0;
  for (int i = 0, sizei = output.size(); i < sizei; i++) {
    Evector e = target[i] - output[i];

    for (int j = 0, sizej = e.size(); j < sizej; j++)
      e[j] *= e[j];

    mse += e.sum();
  }
  return mse / (output.size() * output[0].size());
}
