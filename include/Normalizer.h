//
// Created by Armando Herrera III
// Alot of times it is necessary to scale information to between two intervals
//

#ifndef CPPNNET_NORMALIZER_H
#define CPPNNET_NORMALIZER_H

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <vector>

namespace CppNNet {

  struct NormSettings {
    Eigen::VectorXf xoffset;
    Eigen::VectorXf gain;
    float ymin;
  };

  class Normalizer {
  private:
    NormSettings _settings;
  public:
    Normalizer(std::vector<Eigen::VectorXf> &norm_fig, float ymin, float ymax);

    Normalizer(Eigen::VectorXf &to_norm, Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin);

    Normalizer(Eigen::VectorXf &to_norm, NormSettings settings);

    Normalizer(Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin);

    Normalizer(NormSettings settings);

    void norm(Eigen::VectorXf &to_norm);

    Eigen::VectorXf GetNorm(Eigen::VectorXf &to_norm);

    void reverse(Eigen::VectorXf &to_reverse);

    void reverse(Eigen::VectorXf &to_reverse, NormSettings settings);

    Eigen::VectorXf GetReverse(Eigen::VectorXf &to_reverse);

    void batch_norm(std::vector<Eigen::VectorXf> &to_norm);

    void batch_reverse(std::vector<Eigen::VectorXf> &to_reverse);

    std::vector<Eigen::VectorXf> get_batch_norm(std::vector<Eigen::VectorXf> &to_norm);

    std::vector<Eigen::VectorXf> get_batch_reverse(std::vector<Eigen::VectorXf> &to_reverse);

    NormSettings GetSettings() { return _settings; }

    ~Normalizer() = default;
  };

}

#endif // CPPNNET_NORMALIZER_H
