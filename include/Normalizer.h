#ifndef NORMALIZER_H
#define NORMALIZER_H

/* Alot of times it is necessary to scale information to between two intervals */

#include <Eigen/Core>
#include <vector>

struct NormSettings
{
  Eigen::VectorXf xoffset;
  Eigen::VectorXf gain;
  float ymin;
};

class Normalizer
{
private:
  NormSettings _settings;
public:
  Normalizer(std::vector<Eigen::VectorXf>& norm_fig, float ymin, float ymax);
  Normalizer(Eigen::VectorXf& to_norm, Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin);
  Normalizer(Eigen::VectorXf& to_norm, NormSettings settings);
  Normalizer(Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin);
  Normalizer(NormSettings settings);

  void norm(Eigen::VectorXf& to_norm);

  void reverse(Eigen::VectorXf& to_reverse);
  void reverse(Eigen::VectorXf& to_reverse, NormSettings settings);

  ~Normalizer() {}
};

#endif // NORMALIZER_H
