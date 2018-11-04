//
// Created by Armando Herrera III
// Structure for a neuron layer of a Network
//

#ifndef CPPNNET_NEURAL_LAYER_H
#define CPPNNET_NEURAL_LAYER_H

#include <functional>
#include <memory>
#include <vector>
#include <Eigen/Core>

namespace CppNNet {

  typedef std::function<float(float)> function;
  typedef Eigen::MatrixXf Ematrix;
  typedef Eigen::VectorXf Evector;

  struct norm_data {
    float alpha = 0, beta = 0, epsilon = 0.00001;
  };

  enum activation_function {
    Identity, Logistic, HyperbolicTan, ArcTan, Sin, Gaussian, ReLU, LeakyReLU
  };

  class Neural_Layer : public std::enable_shared_from_this<Neural_Layer> {
  private:
    typedef std::shared_ptr<Neural_Layer> Neural_Ptr;

    // Weights Matrix
    Ematrix _w;
    // Bias Vector
    Evector _b;
    // Activation Function
    activation_function _func = activation_function::Identity;
    function _activ_func = [](float x) { return x; };
    // Pointer to Previous Layer
    Neural_Ptr _prev_layer;

    // Normalization data
    float _alpha, _beta, _epsilon = 0.00001;
    bool _normalize = false;

    // Gets the number of weights and biases
    long parameter_count();

  public:
    // Constructors
    Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer, activation_function func,
                 norm_data *dat = nullptr);

    Neural_Layer(Ematrix Weights, Evector Bias, Neural_Ptr previous_layer, norm_data *dat = nullptr);

    Neural_Layer(Ematrix Weights, Evector Bias, activation_function func, norm_data *dat = nullptr);

    Neural_Layer(Ematrix Weights, Evector Bias, norm_data *dat = nullptr);

    Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer, activation_function func,
                 bool normalize = false);

    Neural_Layer(int nneurons, int ninputs, Neural_Ptr previous_layer, bool normalize = false);

    Neural_Layer(int nneurons, int ninputs, activation_function func, bool normalize = false);

    Neural_Layer(int nneurons, int ninputs, bool normalize = false);

    ~Neural_Layer() = default;

    std::vector<Evector> normalize(const std::vector<Evector> &input);

    Evector feedforward(Evector input);

    std::vector<Evector> feedforward_batch(std::vector<Evector> input);

    // Mean Square Error
    float mse(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Root Mean Square Error
    float rmse(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Mean Absolute Error
    float mae(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Mean Percent Error
    float mpe(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Mean Absolute Percent Error
    float mape(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // coefficient of determination (R squared)
    float r2(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Akaike information criterion (AIC)
    float aic(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Corrected Akaike information criterion (AICc)
    float aicc(const std::vector<Evector> &input, const std::vector<Evector> &target);

    // Bayesian information criterion (BIC)
    float bic(const std::vector<Evector> &input, const std::vector<Evector> &target);

    Ematrix GetWeights() { return _w; }

    Evector GetBiases() { return _b; }

    long GetNInputs() { return _w.cols(); }

    long GetNNeurons() { return _w.rows(); }

    // Function to get vector of pointer to all layers
    std::vector<Neural_Ptr> GetVecPtrs();

    // Gets activation function from activation_function
    function Get_Activation_Function(activation_function func);

    function Get_Activation_Function();

    activation_function Current_Activation_Function();

    // Gets derivative function from activation_function
    function Get_Derivative_Function(activation_function func);

    function Get_Derivative_Function();

    friend class Neural_Trainer;

    friend class MSGD_Neural_Trainer;
  };

}

#endif // CPPNNET_NEURAL_LAYER_H
