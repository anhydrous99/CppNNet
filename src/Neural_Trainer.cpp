#include "Neural_Trainer.h"
#include <iterator>
#include <random>
#include <iostream>

std::vector<int> Neural_Trainer::shuffle_indices(int nindices) {
  std::vector<int> indices;
  indices.reserve(nindices);
  for (int i = 0; i < nindices; ++i)
    indices.push_back(i);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(indices.begin(), indices.end(), g);

  return indices;
}

Neural_Trainer::Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs) {
  _neur_ptrs = neural_ptrs;
  unsigned long M = neural_ptrs.size();
  for (unsigned long m = 0; m < M; m++)
    _daf.push_back(neural_ptrs[m]->Get_Derivative_Function());
}

Neural_Trainer::Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptrs, float learning_rate) :
    Neural_Trainer(neural_ptrs) {
  _learning_rate = learning_rate;
}

Neural_Trainer::Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr) {
  _neur_ptrs = end_neural_ptr->GetVecPtrs();
  unsigned long M = _neur_ptrs.size();
  for (unsigned long m = 0; m < M; m++)
    _daf.push_back(_neur_ptrs[m]->Get_Derivative_Function());
}

Neural_Trainer::Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, float learning_rate) :
    Neural_Trainer(end_neural_ptr) {
  _learning_rate = learning_rate;
}

void Neural_Trainer::train_sample(const Evector &s, const Evector &t) {
  std::vector<Evector> n, a;
  unsigned long M = _neur_ptrs.size();

  // Forward Propagate
  //  a_0 = s
  a.push_back(s);
  for (int m = 1; m <= M; m++) {
    // Calculate n first n = w * a + b
    int mm1 = m - 1;
    std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[mm1];
    Evector n_cur = current_ptr->_w * a[mm1] + current_ptr->_b;
    Evector a_cur(n_cur.size());

    // Calculate new a from n
    for (long i = 0, size = a_cur.size(); i < size; i++)
      a_cur[i] = current_ptr->_activ_func(n_cur[i]);

    // Store both a and n
    n.push_back(std::move(n_cur));
    a.push_back(std::move(a_cur));
  }

  // Backward Propagate
  Evector past_sensitivity;
  for (unsigned long m = M; m >= 1; m--) {
    // Calculate Sensitivities
    unsigned long mm1 = m - 1;
    Evector sensitivity(n[mm1].size());
    std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[mm1];
    if (m == M) {
      // Calculate first sensitivities
      for (long i = 0, size = n[mm1].size(); i < size; i++)
        sensitivity[i] = -2 * _daf[mm1](n[mm1][i]) * (t[i] - a[m][i]);
    } else {
      for (long i = 0, sizei = n[mm1].size(); i < sizei; i++) {
        sensitivity[i] = 0.0;
        for (long j = 0, sizej = past_sensitivity.size(); j < sizej; j++)
          sensitivity[i] += _neur_ptrs[m]->_w(j, i) * past_sensitivity[j];
        sensitivity[i] *= _daf[mm1](n[mm1][i]);
      }
    }

    // Calculate new weights
    current_ptr->_w -= _learning_rate * sensitivity * a[mm1].matrix().transpose();

    // Calculate new bias
    current_ptr->_b -= _learning_rate * sensitivity;

    // Store Sensitivity for future use
    past_sensitivity = std::move(sensitivity);
  }
}

void Neural_Trainer::train_batch(const std::vector<Evector> &s, const std::vector<Evector> &t) {
  unsigned long Q = s.size(),
      M = _neur_ptrs.size();
  // Temporary vector
  std::vector<int> idxs = shuffle_indices(Q);
  // Declare and initialize sa and s matrix and vector
  std::vector<Ematrix> sa_sum;
  std::vector<Evector> s_sum;
  for (unsigned long m = 0; m < M; m++) {
    s_sum.emplace_back(Evector::Zero(_neur_ptrs[m]->_b.size()));
    sa_sum.emplace_back(Ematrix::Zero(_neur_ptrs[m]->_w.rows(), _neur_ptrs[m]->_w.cols()));
  }

  for (unsigned long q = 0; q < Q; q++) {
    std::vector<Evector> aq, nq;
    int idx = idxs[q];
    aq.push_back(s[idx]);
    for (unsigned long m = 1; m <= M; m++) {
      unsigned long mm1 = m - 1;
      // Calculate n = w * a + b
      std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[mm1];
      Evector n_cur = current_ptr->_w * aq[mm1] + current_ptr->_b;
      Evector a_cur(n_cur.size());

      // Calculate new a from n
      for (long i = 0, size = a_cur.size(); i < size; i++)
        a_cur[i] = current_ptr->_activ_func(n_cur[i]);

      // Store both a and n
      nq.push_back(std::move(n_cur));
      aq.push_back(std::move(a_cur));
    }

    // Backward Propagate
    Evector past_sensitivity;
    for (unsigned long m = M; m >= 1; m--) {
      // Calculate Sensitivities
      unsigned long mm1 = m - 1;
      Evector sensitivity = Evector::Zero(nq[mm1].size());
      std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[mm1];
      if (m == M) {
        // Calculate first sensitivities
        for (long i = 0, size = nq[mm1].size(); i < size; i++)
          sensitivity[i] = -2 * _daf[mm1](nq[mm1][i]) * (t[idx][i] - aq[m][i]);
      } else {
        for (long i = 0, size = nq[mm1].size(); i < size; i++) {
          for (long j = 0, sizej = past_sensitivity.size(); j < sizej; j++)
            sensitivity[i] += _neur_ptrs[m]->_w(j, i) * past_sensitivity[j];
          sensitivity[i] *= _daf[mm1](nq[mm1][i]);
        }
      }

      // Sum s * a^t and s
      sa_sum[mm1] += sensitivity * aq[mm1].matrix().transpose();
      s_sum[mm1] += sensitivity;

      // Store sensitivity for future use
      past_sensitivity = std::move(sensitivity);
    }
  }

  for (unsigned long m = 0; m < M; m++) {
    std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[m];
    float aq = _learning_rate / Q;
#pragma omp critical (sgd_train_batch)
    {
      // Calculate new weights
      current_ptr->_w -= aq * sa_sum[m];
      // Calculate new biases
      current_ptr->_b -= aq * s_sum[m];
    }
  }
}

void Neural_Trainer::train_minibatch(const std::vector<Evector> &s, const std::vector<Evector> &t,
                                     unsigned long batch_size) {
  // Get begin iterator and end iterator for s
  auto s_begin(s.cbegin());
  auto s_end(s.cend());

  // Get begin iterator and end iterator for t
  auto t_begin(t.cbegin());
  auto t_end(t.cend());

  auto n_iterations = (unsigned long) std::floor((double) s.size() / (double) batch_size);

#pragma omp parallel for
  for (unsigned long i = 0; i < n_iterations; i++) {
    // s slice iterators
    auto s_slice_start(s_begin + i * batch_size);
    auto s_slice_end((i != n_iterations - 1) ? s_begin + (i + 1) * batch_size : s_end);

    // t slice iterators
    auto t_slice_start(t_begin + i * batch_size);
    auto t_slice_end((i != n_iterations - 1) ? t_begin + (i + 1) * batch_size : t_end);

    // Create vectors from slices
    std::vector<Evector> s_slice(s_slice_start, s_slice_end);
    std::vector<Evector> t_slice(t_slice_start, t_slice_end);

    // Train
    train_batch(s_slice, t_slice);
  }
}
