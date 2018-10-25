//
// Created by Armando Herrera III on 10/24/18.
//

#include "MSGD_Neural_Trainer.h"

MSGD_Neural_Trainer::MSGD_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr,
                                         std::vector<function> derv_fun) : Neural_Trainer(neural_ptr, derv_fun) {
  _init();
}

MSGD_Neural_Trainer::MSGD_Neural_Trainer(std::vector<std::shared_ptr<Neural_Layer>> neural_ptr,
                                         std::vector<function> derv_fun, learning_momentum lrm) : Neural_Trainer(
    neural_ptr, derv_fun, lrm.learning_rate) {
  _momentum_constant = lrm.momentum;
  _init();
}

MSGD_Neural_Trainer::MSGD_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun) :
    Neural_Trainer(end_neural_ptr, derv_fun) {
  _init();
}

MSGD_Neural_Trainer::MSGD_Neural_Trainer(std::shared_ptr<Neural_Layer> end_neural_ptr, std::vector<function> derv_fun,
                                         learning_momentum lrm) : Neural_Trainer(end_neural_ptr, derv_fun,
                                                                                 lrm.learning_rate) {
  _momentum_constant = lrm.momentum;
  _init();
}

void MSGD_Neural_Trainer::_init() {
  unsigned long M = _neur_ptrs.size();
  for (unsigned long m = 0; m < M; m++) {
    _past_weights.emplace_back(Ematrix::Zero(_neur_ptrs[m]->_w.rows(), _neur_ptrs[m]->_w.cols()));
    _past_biases.emplace_back(Evector::Zero(_neur_ptrs[m]->_b.size()));
  }
}

void MSGD_Neural_Trainer::train_sample(const Evector &s, const Evector &t) {
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
    _past_weights[mm1] =
        _momentum_constant * _past_weights[mm1] - _learning_rate * sensitivity * a[mm1].matrix().transpose();
    current_ptr->_w += _past_weights[mm1];

    // Calculate new bias
    _past_biases[mm1] = _momentum_constant * _past_biases[mm1] - _learning_rate * sensitivity;
    current_ptr->_b += _past_biases[mm1];

    // Store Sensitivity for future use
    past_sensitivity = std::move(sensitivity);
  }
}

#include <iostream>

void MSGD_Neural_Trainer::train_batch(const std::vector<Evector> &s, const std::vector<Evector> &t) {
  unsigned long Q = s.size(),
      M = _neur_ptrs.size();
  // Temporary vectors
  std::vector<int> idxs = shuffle_indices(Q);
  std::vector<Ematrix> past_weights;
  std::vector<Evector> past_biases;
  for (unsigned long m = 0; m < M; m++) {
    past_weights.emplace_back(Ematrix::Zero(_neur_ptrs[m]->_w.rows(), _neur_ptrs[m]->_w.cols()));
    past_biases.emplace_back(Evector::Zero(_neur_ptrs[m]->_b.size()));
  }

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

      past_sensitivity = std::move(sensitivity);
    }
  }

  for (unsigned long m = 0; m < M; m++) {
    std::shared_ptr<Neural_Layer> current_ptr = _neur_ptrs[m];
    float aq = _learning_rate / Q;
    // Calculate new weights
    past_weights[m] = _momentum_constant * past_weights[m] - aq * sa_sum[m];
    current_ptr->_w += past_weights[m];
    // Calculate new biases
    past_biases[m] = _momentum_constant * past_biases[m] - aq * s_sum[m];
    current_ptr->_b += past_biases[m];
  }
}