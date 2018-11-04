//
// Created by Armando Herrera III
// A small repository of activation functions and their derivatives
//

#ifndef CPPNNET_ACTIVATION_FUNCTIONS_H
#define CPPNNET_ACTIVATION_FUNCTIONS_H

#include <functional>
#include <cmath>

namespace CppNNet {
// Identity Function
  std::function<float(float)> Identity_Function = [](float x) { return x; };
  std::function<float(float)> Identity_Function_D = [](float x) { return 1; };

// Logistic aka Sigmoid or Soft step
  std::function<float(float)> Logistic_Function = [](float x) { return 1 / (1 + exp(-x)); };
  std::function<float(float)> Logistic_Function_D = [](float x) {
    float ex = exp(x);
    return ex / pow(1 + ex, 2);
  };

// Hyperbolic Tangent
  std::function<float(float)> HyperbolicTan_Function = [](float x) { return tanh(x); };
  std::function<float(float)> HyperbolicTan_Function_D = [](float x) { return 1 - pow(HyperbolicTan_Function(x), 2); };
// ArcTangent
  std::function<float(float)> ArcTan_Function = [](float x) { return atan(x); };
  std::function<float(float)> ArcTan_Function_D = [](float x) { return 1 / (pow(x, 2) + 1); };

// Sinusoid
  std::function<float(float)> Sin_Function = [](float x) { return sin(x); };
  std::function<float(float)> Sin_Function_D = [](float x) { return cos(x); };

// Gaussian
  std::function<float(float)> Gaussian_Function = [](float x) { return exp(-pow(x, 2)); };
  std::function<float(float)> Gaussian_Function_D = [](float x) { return -2 * x * exp(-pow(x, 2)); };
}

#endif // CPPNNET_ACTIVATION_FUNCTIONS_H
