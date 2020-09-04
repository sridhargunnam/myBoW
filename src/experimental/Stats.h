//
// Created by srigun on 9/3/20.
//

#ifndef MYBOW_SRC_EXPERIMENTAL_STATS_H_
#define MYBOW_SRC_EXPERIMENTAL_STATS_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <cassert>

template <typename T>
class Stats
{
  std::vector<std::vector<T>> all_hists_normalized_;
  std::vector<T> mean_;
  std::vector<T> variance_;
  std::vector<T> all_variance;

public:
  explicit Stats( std::vector<std::vector<T>>& all_hists_normalized): all_hists_normalized_{all_hists_normalized}{
    ComputeStats();
  };

    void ComputeStats(){
    assert(all_hists_normalized_.size() != 0);
    assert(all_hists_normalized_[0].size() != 0);

    std::vector<T> all_sum;
    all_sum.resize(all_hists_normalized_[0].size());
    std::fill(all_sum.begin(), all_sum.end(), 0);
    for(auto hi : all_hists_normalized_){
      std::transform(hi.cbegin(), hi.cend(), all_sum.begin(), all_sum.begin(), [](const auto& first, auto& second){second += first; return second;} );
    }

    PrintVecContainer("all_sum", all_sum);
    std::vector<T> all_mean{all_sum};
    std::transform(all_mean.begin(), all_mean.end(), all_mean.begin(), [&](const auto& first){return first/static_cast<T>(all_hists_normalized_.size());});
    PrintVecContainer("all_mean", all_mean);

    auto sum_of_all_sum = std::accumulate(all_sum.cbegin(), all_sum.cend(), 0.0);
    auto sum_of_all_mean = std::accumulate(all_mean.cbegin(), all_mean.cend(), 0.0);
    std::cout << "Sum of all_sum  = " << sum_of_all_sum  << "\n";
    std::cout << "Sum of all_mean = " << sum_of_all_mean << "\n";

    all_variance.resize(all_hists_normalized_[0].size());
    std::fill(all_variance.begin(), all_variance.end(), 0);
    std::vector<T> diffs;
    diffs.resize(all_hists_normalized_[0].size());

    // Calculating variance, don't worry about the below mess. Clean up when you can
    for(auto hi : all_hists_normalized_){
      std::copy(diffs.begin(), diffs.end(), all_mean.begin());
      std::transform(hi.cbegin(), hi.cend(), diffs.begin(), diffs.begin(),
        [&](const auto& first, auto& second){second = std::pow((first-second),2); return second;} );
      std::transform(diffs.cbegin(), diffs.cend(), all_variance.begin(), all_variance.begin(),
        [](const auto& first, auto& second) {second += first; return second;});
    }
    
    for_each(all_variance.begin(), all_variance.end(), [&](auto& val){val = sqrt( val/static_cast<T>(all_hists_normalized_.size() ));});
    PrintVecContainer("all_variance", all_variance);

  };

  void PrintVecContainer(const std::string& name, const std::vector<T>& vec){
    std::cout <<  name << " = \n";
    std::for_each(vec.begin(), vec.end(), [](const T val){std::cout << val << " ";});
    std::cout << "\n";
  }


};


#endif//MYBOW_SRC_EXPERIMENTAL_STATS_H_
