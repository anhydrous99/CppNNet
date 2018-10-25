//
// Created by Armando Herrera III
// Alot of times it is necessary to scale information to between two intervals
//

#ifndef CPPNNET_NORMALIZER_H
#define CPPNNET_NORMALIZER_H

#include <Eigen/Core>
#include <vector>

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

#endif // CPPNNET_NORMALIZER_H
