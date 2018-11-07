#include "Neural_Layer.h"
#include "Activation_Functions.h"

#include <algorithm>
#include <random>
#include <iterator>
#include <cmath>

CppNNet::Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer, activation_function func,
                                    norm_data *dat) :
    Neural_Layer(std::move(Weights), std::move(Bias), std::move(previous_layer), dat) {
  _func = func;
  _activ_func = Get_Activation_Function(func);
}

CppNNet::Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer, norm_data *dat) :
    Neural_Layer(std::move(Weights), std::move(Bias), dat) {
  _prev_layer = std::move(previous_layer);
}

CppNNet::Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, activation_function func, norm_data *dat) :
    Neural_Layer(std::move(Weights), std::move(Bias), dat) {
  _func = func;
  _activ_func = Get_Activation_Function(func);
}

CppNNet::Neural_Layer::Neural_Layer(Ematrix Weights, Evector Bias, norm_data *dat) {
  _w = std::move(Weights);
  _b = std::move(Bias);

  if (dat != nullptr) {
    _normalize = true;
    _alpha = dat->alpha;
    _beta = dat->beta;
  }
}

CppNNet::Neural_Layer::Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer, activation_function func,
                                    bool normalize) :
    Neural_Layer(nneurons, ninputs, std::move(previous_layer), normalize) {
  _func = func;
  _activ_func = Get_Activation_Function(func);
}

CppNNet::Neural_Layer::Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer, bool normalize) :
    Neural_Layer(nneurons, ninputs, normalize) {
  _prev_layer = std::move(previous_layer);
}

CppNNet::Neural_Layer::Neural_Layer(int nneurons, int ninputs, activation_function func, bool normalize) :
    Neural_Layer(nneurons, ninputs, normalize) {
  _func = func;
  _activ_func = Get_Activation_Function(func);
}

CppNNet::Neural_Layer::Neural_Layer(int nneurons, int ninputs, bool normalize) : _w(nneurons, ninputs), _b(nneurons) {
  // Obtain seed
  std::random_device rd;
  // Standard merseen_twister_engine
  std::mt19937 gen(rd());
  // Initialization using Gaussian distribution
  std::normal_distribution<float> dis(0.0, 1.0);

  // Helps with variance
  float xiv = 1;
  if (_func == activation_function::LeakyReLU || _func == activation_function::ReLU)
    xiv = std::sqrt(static_cast<float>(2.0) / ninputs);
  else if (_func == activation_function::HyperbolicTan)
    xiv = std::sqrt(static_cast<float>(1.0) / (ninputs + nneurons));

  // Fill the weights with random numbers
  for (long i = 0, size = _w.size(); i < size; i++)
    *(_w.data() + i) = dis(gen) * xiv;

  // Fill the bias with random numbers
  for (long i = 0, size = _b.size(); i < size; i++)
    *(_b.data() + i) = dis(gen) * xiv;

  if (normalize) {
    _normalize = true;
    _alpha = dis(gen);
    _beta = dis(gen);
  }
}

// Utility function


std::vector<CppNNet::Evector> CppNNet::Neural_Layer::normalize(const std::vector<Evector> &input) {
  unsigned long n = input.size();
  // Mini-batch mean
  Evector mu = Evector::Zero(input[0].size());
  for (auto &it : input)
    mu += it;
  mu /= n;

  // Mini-batch variance
  Evector sigma2 = Evector::Zero(input[0].size());
  for (auto &it : input)
    sigma2 += (it - mu).array().square().matrix();
  sigma2 /= n;

  // Normalize
  std::vector<Evector> output;
  for (auto &it : input)
    output.emplace_back((_alpha * (it - mu).array() / (_epsilon + sigma2.array()) + _beta).matrix());
  return output;
}


CppNNet::Evector CppNNet::Neural_Layer::feedforward(Evector input) {
  Evector a = (_prev_layer) ? _prev_layer->feedforward(input) : input;
  Evector n = _w * a + _b;

  for (long i = 0, size = n.size(); i < size; i++)
    n[i] = _activ_func(n[i]);

  return n;
}

std::vector<CppNNet::Evector> CppNNet::Neural_Layer::feedforward_batch(std::vector<Evector> input) {
  std::vector<Evector> a = (_prev_layer) ? _prev_layer->feedforward_batch(input) : input;
  std::vector<Evector> n;
  for (auto &it : a)
    n.emplace_back(_w * it + _b);

  if (_normalize)
    n = normalize(n);

  for (auto &it : n) {
    for (long i = 0, size = it.size(); i < size; i++)
      it[i] = _activ_func(it[i]);
  }

  return n;
}

std::vector<std::shared_ptr<CppNNet::Neural_Layer>> CppNNet::Neural_Layer::GetVecPtrs() {
  std::vector<std::shared_ptr<Neural_Layer>> output;

  if (_prev_layer)
    output = _prev_layer->GetVecPtrs();
  output.push_back(shared_from_this());
  return output;
}

// Mean Square Error
float CppNNet::Neural_Layer::mse(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
float CppNNet::Neural_Layer::rmse(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  return std::sqrt(mse(input, target));
}

// Mean absolute error
float CppNNet::Neural_Layer::mae(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
float CppNNet::Neural_Layer::mpe(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
float CppNNet::Neural_Layer::mape(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
CppNNet::Evector CppNNet::Neural_Layer::r2(const std::vector<Evector> &input, const std::vector<Evector> &target) {
  std::vector<Evector> output = feedforward_batch(input);
  long sizei = target.size(),
      sizej = target[0].size();

  Evector y_mean = Evector::Zero(sizej);

  for (long i = 0; i < sizei; i++)
    y_mean += target[i];
  y_mean /= sizei;

  Evector sstot = Evector::Zero(sizej);
  Evector ssres = Evector::Zero(sizej);

  for (long i = 0; i < sizei; i++) {
    sstot += (target[i] - y_mean).array().square().matrix();
    ssres += (target[i] - output[i]).array().square().matrix();
  }

  return Evector::Ones(sizej) - (ssres.array() / sstot.array()).matrix();
}

float CppNNet::Neural_Layer::r2_avg(const std::vector<CppNNet::Evector> &input,
                                    const std::vector<CppNNet::Evector> &target) {
  Evector pass = r2(input, target);
  return pass.sum() / pass.size();
}

// Returns number of weights and biases for the layer and all layers prior
long CppNNet::Neural_Layer::parameter_count() {
  long a = (_prev_layer) ? _prev_layer->parameter_count() : 0;
  return _w.size() + _b.size() + a;
}

// Akaike information criterion (AIC)
float CppNNet::Neural_Layer::aic(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
float CppNNet::Neural_Layer::aicc(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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
float CppNNet::Neural_Layer::bic(const std::vector<Evector> &input, const std::vector<Evector> &target) {
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


CppNNet::function CppNNet::Neural_Layer::Get_Activation_Function(activation_function func) {
  switch (func) {
    default:
    case activation_function::Identity:
      return Identity_Function;
    case activation_function::Logistic:
      return Logistic_Function;
    case activation_function::HyperbolicTan:
      return HyperbolicTan_Function;
    case activation_function::ArcTan:
      return ArcTan_Function;
    case activation_function::Sin:
      return Sin_Function;
    case activation_function::Gaussian:
      return Gaussian_Function;
    case activation_function::ReLU:
      return ReLU_Function;
    case activation_function::LeakyReLU:
      return LeakyReLU_Function;
  }
}

CppNNet::function CppNNet::Neural_Layer::Get_Activation_Function() {
  return Get_Activation_Function(_func);
}

CppNNet::activation_function CppNNet::Neural_Layer::Current_Activation_Function() {
  return _func;
}

CppNNet::function CppNNet::Neural_Layer::Get_Derivative_Function(activation_function func) {
  switch (func) {
    default:
    case activation_function::Identity:
      return Identity_Function_D;
    case activation_function::Logistic:
      return Logistic_Function_D;
    case activation_function::HyperbolicTan:
      return HyperbolicTan_Function_D;
    case activation_function::ArcTan:
      return ArcTan_Function_D;
    case activation_function::Sin:
      return Sin_Function_D;
    case activation_function::Gaussian:
      return Gaussian_Function_D;
    case activation_function::ReLU:
      return ReLU_Function_D;
    case activation_function::LeakyReLU:
      return LeakyReLU_Function_D;
  }
}

CppNNet::function CppNNet::Neural_Layer::Get_Derivative_Function() {
  return Get_Derivative_Function(_func);
}