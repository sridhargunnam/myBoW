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
#include <iomanip>


template <typename T>
void PrintVecContainer(const std::string& name, const std::vector<T>& vec, int precision_bits = 3){
  std::cout <<  name << " = \n";
  std::cout << std::fixed;
  std::cout << std::setprecision(precision_bits);
  std::for_each(vec.begin(), vec.end(), [](const T val){std::cout << val << ", ";});
  std::cout << "\n";
}


template <typename T>
class Stats
{
  std::vector<std::vector<T>> all_hists_normalized_;
  std::vector<T> all_mean_;
  std::vector<T> variance_;
  std::vector<T> all_variance_;
  std::vector<std::string> file_list_;

public:
  explicit Stats( std::vector<std::vector<T>>& all_hists_normalized,
    const std::vector<std::string>& file_list):
                                                 all_hists_normalized_{all_hists_normalized},
                                                 file_list_(file_list){
    ComputeStats();
  };

  std::vector<T> getVariance(){
    return all_variance_;
  }
  std::vector<T> getMean(){
    return all_mean_;
  }

    void ComputeStats(){
      int index = 0;

    assert(all_hists_normalized_.size() != 0);
    assert(all_hists_normalized_[0].size() != 0);

    std::vector<T> all_sum;
    all_sum.resize(all_hists_normalized_[0].size());
    std::fill(all_sum.begin(), all_sum.end(), 0);
    for(auto hi : all_hists_normalized_){
      std::transform(hi.cbegin(), hi.cend(), all_sum.begin(), all_sum.begin(), [](const auto& first, auto& second){second += first; return second;} );
    }

    PrintVecContainer<T>("all_sum", all_sum);
    all_mean_.resize(all_sum.size());
    std::copy(all_sum.begin(), all_sum.end(), all_mean_.begin());
    std::transform(all_mean_.begin(), all_mean_.end(), all_mean_.begin(), [&](const auto& first){return first/static_cast<T>(all_hists_normalized_.size());});
    PrintVecContainer<T>("all_mean", all_mean_);

    auto sum_of_all_sum = std::accumulate(all_sum.cbegin(), all_sum.cend(), 0.0);
    auto sum_of_all_mean = std::accumulate(all_mean_.cbegin(), all_mean_.cend(), 0.0);
    std::cout << "Sum of all_sum  = " << sum_of_all_sum  << "\n";
    std::cout << "Sum of all_mean = " << sum_of_all_mean << "\n";

    all_variance_.resize(all_hists_normalized_[0].size());
    std::fill(all_variance_.begin(), all_variance_.end(), 0);
    std::vector<T> diffs;
    diffs.resize(all_hists_normalized_[0].size());

    // Calculating variance, don't worry about the below mess. Clean up when you can
    for(auto hi : all_hists_normalized_){
      //std::cout << "Current image filename is " << file_list_[index] << "\n";
      //PrintVecContainer(std::to_string(index), hi);
      ++index;
      std::copy(all_mean_.begin(), all_mean_.end(), diffs.begin());
      std::transform(hi.cbegin(), hi.cend(), diffs.begin(), diffs.begin(),
        [&](const auto& first, auto& second){second = std::pow((first-second),2); return second;} );
      std::transform(diffs.cbegin(), diffs.cend(), all_variance_.begin(), all_variance_.begin(),
        [](const auto& first, auto& second) {second += first; return second;});
    }
    for_each(all_variance_.begin(), all_variance_.end(), [&](auto& val){val = sqrt( val/static_cast<T>(all_hists_normalized_.size() ));});
    PrintVecContainer<T>("all_variance_", all_variance_);

  };



};


#endif//MYBOW_SRC_EXPERIMENTAL_STATS_H_
