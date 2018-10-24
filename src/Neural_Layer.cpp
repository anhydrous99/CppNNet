#include "Neural_Layer.h"

#include <algorithm>
#include <random>
#include <cmath>

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
  for (long i = 0, size = _w.size(); i < size; i++)
    *(_w.data() + i) = dis(gen);

  // Fill the bias with random numbers
  for (long i = 0, size = _b.size(); i < size; i++)
    *(_b.data() + i) = dis(gen);
}

Evector Neural_Layer::feedforward(Evector input) {
  Evector a = (_prev_layer) ? _prev_layer->feedforward(input) : input;
  Evector n = _w * a + _b;

  for (long i = 0, size = n.size(); i < size; i++)
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

// Mean Square Error
float Neural_Layer::mse(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);

  float mse = 0.0;
  for (unsigned long i = 0, sizei = output.size(); i < sizei; i++) {
    Evector e = target[i] - output[i];

    for (long j = 0, sizej = e.size(); j < sizej; j++)
      e[j] *= e[j];

    mse += e.sum();
  }
  return mse / (output.size() * output[0].size());
}

// Root Mean Square Error
float Neural_Layer::rmse(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  return std::sqrt(mse(input, target));
}

// Mean absolute error
float Neural_Layer::mae(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);
  float sum = 0.0;
  long sizei = target.size(),
      sizej = target[0].size();
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      sum += std::abs(target[i][j] - output[i][j]);
  }
  return sum / (sizei * sizej);
}

// Mean Percent Error
float Neural_Layer::mpe(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);
  float sum = 0.0;
  long sizei = target.size(),
      sizej = target[0].size();
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      sum += (1 - output[i][j] / target[i][j]);
  }
  return 100 * sum / (sizei * sizej);
}

// Mean Absolute Percentage Error
float Neural_Layer::mape(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);
  float sum = 0.0;
  long sizei = target.size(),
      sizej = target[0].size();
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      sum += std::abs(1 - output[i][j] / target[i][j]);
  }
  return 100 * sum / (sizei * sizej);
}

// coefficient of determination (R squared)
float Neural_Layer::r2(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);
  long sizei = target.size(),
      sizej = target[0].size();
  auto n = static_cast<float>(sizei * sizej);

  float y_mean = std::accumulate(input.begin(), input.end(), 0,
                                 [](float a, const Evector &b) {
                                   return a + b.sum();
                                 }) / n;

  float sstot = 0.0;
  float ssres = 0.0;

  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++) {
      sstot += std::pow(input[i][j] - y_mean, 2);
      ssres += std::pow(input[i][j] - output[i][j], 2);
    }
  }

  return 1 - (ssres / sstot);
}

// Returns number of weights and biases for the layer and all layers prior
long Neural_Layer::parameter_count() {
  long a = (_prev_layer) ? _prev_layer->parameter_count() : 0;
  return _w.size() + _b.size() + a;
}

// Akaike information criterion (AIC)
float Neural_Layer::aic(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  long p = parameter_count();
  std::vector<Evector> output = feedforward_batch(input);
  long sizei = target.size(),
      sizej = target[0].size();

  float SSE = 0.0;
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      SSE += std::pow(target[i][j] - output[i][j], 2);
  }

  return sizei * std::log(SSE / sizei) + 2 * p;
}

// Corrected Akaike information criterion (AICc)
float Neural_Layer::aicc(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  long p = parameter_count();
  std::vector<Evector> output = feedforward_batch(input);
  long sizei = target.size(),
      sizej = target[0].size();

  float SSE = 0.0;
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      SSE += std::pow(target[i][j] - output[i][j], 2);
  }

  return sizei * (static_cast<float>(sizei + p) / static_cast<float>(-2 + sizei - p) + std::log(SSE / sizei));
}

// Bayesian information criterion (BIC)
float Neural_Layer::bic(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  long p = parameter_count();
  std::vector<Evector> output = feedforward_batch(input);
  long sizei = target.size(),
      sizej = target[0].size();

  float SSE = 0.0;
  for (long i = 0; i < sizei; i++) {
    for (long j = 0; j < sizej; j++)
      SSE += std::pow(target[i][j] - output[i][j], 2);
  }

  return sizei * std::log(SSE / sizei) + p * std::log(static_cast<float>(sizei));
}