#include "Neural_Trainer.h"

#include <algorithm>
#include <random>
#include <iterator>

std::vector<int> shuffle_indices(int nindices)
{
  std::vector<int> indices;
  indices.reserve(nindices);
  for (int i = 0; i < nindices; ++i)
    indices.push_back(i);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(indices.begin(), indices.end(), g);

  return indices;
}

Neural_Trainer::Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs, std::vector<function> derv_funs)
{
  _neur_ptrs = neural_ptrs;
  _daf = derv_funs;
}

Neural_Trainer::Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs, 
    std::vector<function> derv_funs, float learning_rate) : Neural_Trainer(neural_ptrs, derv_funs)
{
  _learning_rate = learning_rate;
}

Neural_Trainer::Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_funs)
{
  _neur_ptrs = end_neural_ptr->GetVecPtrs();
  _daf = derv_funs;
}

Neural_Trainer::Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, 
    std::vector<function> derv_funs, float learning_rate) :
  Neural_Trainer(end_neural_ptr, derv_funs)
{
  _learning_rate = learning_rate;
}

void Neural_Trainer::train_sample(Evector s, Evector t)
{
  std::vector<Evector> n, a;

  // Forward Propagate
  a.push_back(s);
  for (int i = 1, size = _neur_ptrs.size(); i <= size; i++)
  {
    std::shared_ptr<Neural_Layer> cur_ptr = _neur_ptrs[i-1];
    Evector ni = cur_ptr->_w * a[i-1] + cur_ptr->_b;
    Evector ai = ni;

    for (int j = 0, sizej = ai.size(); j < sizej; j++)
      ai[j] =  cur_ptr->_activ_func(ai[j]);

    n.push_back(ni);
    a.push_back(ai);
  }

  // Backward Propagate
  int si = _neur_ptrs.size();
  Evector Sp1;
  for (int i = si; i >= 1; i--)
  {
    // Calculate Sensitivities
    int i1 = i-1;
    Evector sen(n[i1].size());
    std::shared_ptr<Neural_Layer> cur_ptr = _neur_ptrs[i1];
    if (i == si)
    {
      Evector e = t - s;
      for (int j = 0, sizej = n[i1].size(); j < sizej; j++)
        sen[j] = - 2 * _daf[i1](n[i1][j]) * e[j];
    }
    else
    {
      for (int j = 0, sizej = sen.size(); j < sizej; j++)
      {
        float sum = 0;
        for (int k = 0, sizek = Sp1.size(); k < sizek; k++)
          sum += _daf[i1](n[i1][j]) * _neur_ptrs[i]->_w(k,j) * Sp1[k];
        sen[j] = sum;
      }
    }
    Sp1 = sen;

    // Calculate New weights
    Ematrix sa(sen.size(), a[i1].size());
    for (int j = 0, sizej = sen.size(); j < sizej; j++)
      for (int k = 0, sizek = a[i1].size(); k < sizek; k++)
        sa(j,k) = sen[j] * a[i1][k];

    cur_ptr->_w = cur_ptr->_w - _learning_rate * sa;

    // Calculate New Biases
    cur_ptr->_b = cur_ptr->_b - _learning_rate * sen;
  }
}
