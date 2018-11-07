//
// Created by ubrdog on 11/7/18.
//

#ifndef CPPNNET_UTIL_H
#define CPPNNET_UTIL_H

#include <iterator>
#include <random>

namespace CppNNet {
  template<class RandomIT, class URBG>
  void double_shuffle(RandomIT first1, RandomIT last1, RandomIT first2, URBG &&g) {
    typedef typename std::iterator_traits<RandomIT>::difference_type diff_t;
    typedef std::uniform_int_distribution<diff_t> distr_t;
    typedef typename distr_t::param_type param_t;

    distr_t D;
    diff_t n = last1 - first1;

    for (diff_t i = n - 1; i > 0; --i) {
      using std::swap;
      auto dd = D(g, param_t(0, i));
      swap(first1[i], first1[dd]);
      swap(first2[i], first2[dd]);
    }
  }

  template<class RandomIT>
  void double_shuffle(RandomIT first1, RandomIT last1, RandomIT first2) {
    std::random_device rd;
    std::mt19937 g(rd());

    double_shuffle(first1, last1, first2, g);
  }
}

#endif //CPPNNET_UTIL_H
