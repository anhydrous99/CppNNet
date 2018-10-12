#include "Normalizer.h"

Normalizer::Normalizer(std::vector<Eigen::VectorXf>& xlist, float ymin, float ymax)
{
  int isize = xlist.size();
  int jsize = xlist[0].size();
  Eigen::VectorXf gain(jsize);
  Eigen::VectorXf xoffset(jsize);

  float diff = ymax - ymin;
  for (int j = 0; j < jsize; j++)
  {
    float xmin = 0, xmax = 0;
    for (int i = 0; i < isize; i++)
    {
      float cur = xlist[i][j];
      if (xmin < cur) xmin = cur;
      if (xmax > cur) xmax = cur;
    }
    xoffset[j] = xmin;
    gain[j] = diff / (xmax - xmin);
  }

  _settings.xoffset = xoffset;
  _settings.gain = gain;
  _settings.ymin = ymin;
}

Normalizer::Normalizer(Eigen::VectorXf& to_norm, Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin) :
  Normalizer(xoffset, gain, ymin)
{
  norm(to_norm);
}

Normalizer::Normalizer(Eigen::VectorXf& to_norm, NormSettings settings)
{
  _settings = settings;
  for (int i = 0, size = to_norm.size(); i < size; i++)
    to_norm[i] = (to_norm[i] - settings.xoffset[i]) * settings.gain[i] + settings.ymin;
}

Normalizer::Normalizer(Eigen::VectorXf xoffset, Eigen::VectorXf gain, float ymin)
{
  _settings.ymin = ymin;
  _settings.xoffset = xoffset;
  _settings.gain = gain;
}

Normalizer::Normalizer(NormSettings settings)
{ _settings = settings; }

void Normalizer::norm(Eigen::VectorXf& to_norm)
{
  for (int i = 0, size = to_norm.size(); i < size; i++)
    to_norm[i] = (to_norm[i] - _settings.xoffset[i]) * _settings.gain[i] + _settings.ymin;
}

void Normalizer::reverse(Eigen::VectorXf& to_reverse)
{
  for (int i = 0, size = to_reverse.size(); i < size; i++)
    to_reverse[i] = ((to_reverse[i] - _settings.ymin) / _settings.gain[i]) + _settings.xoffset[i];
}

void Normalizer::reverse(Eigen::VectorXf& to_reverse, NormSettings settings)
{
  _settings = settings;
  reverse(to_reverse);
}
