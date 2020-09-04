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

class Stats
{
  std::vector<std::vector<double>> all_hists_normalized_;
  std::vector<double> mean_;
  std::vector<double> variance_;
  std::vector<double> all_variance;

public:
  explicit Stats( std::vector<std::vector<double>>& all_hists_normalized): all_hists_normalized_{all_hists_normalized}{
  };

    void ComputeStats(){
    assert(all_hists_normalized_.size() != 0);
    assert(all_hists_normalized_[0].size() != 0);

    std::vector<double> all_sum;
    all_sum.resize(all_hists_normalized_[0].size());
    std::fill(all_sum.begin(), all_sum.end(), 0);
    for(auto hi : all_hists_normalized_){
      std::transform(hi.cbegin(), hi.cend(), all_sum.begin(), all_sum.end(), [](const auto& first, auto& second){second += first; return second;} );
    }

    PrintVecContainer("all_sum", all_sum);
    std::vector<double> all_mean{all_sum};
    std::transform(all_mean.begin(), all_mean.end(), all_mean.begin(), [&](const auto& first){return first/static_cast<double>(all_hists_normalized_.size());});
    PrintVecContainer("all_mean", all_mean);


    all_variance.resize(all_hists_normalized_[0].size());
    std::fill(all_variance.begin(), all_variance.end(), 0);
    std::vector<double> diffs;
    diffs.resize(all_hists_normalized_[0].size());

    // Calculating variance, don't worry about the below mess. Clean up when you can
    for(auto hi : all_hists_normalized_){
      std::copy(diffs.begin(), diffs.end(), all_mean.begin());
      std::transform(hi.cbegin(), hi.cend(), diffs.begin(), diffs.end(),
        [&](const auto& first, auto& second){second = std::pow((first-second),2); return second;} );
      std::transform(diffs.cbegin(), diffs.cend(), all_variance.begin(), all_variance.end(),
        [](const auto& first, auto& second) {second += first; return second;});
    }
    
    for_each(all_variance.begin(), all_variance.end(), [&](auto& val){val = sqrt( val/static_cast<double>(all_hists_normalized_.size() ));});
    PrintVecContainer("all_variance", all_variance);

  };

  void PrintVecContainer(const std::string& name, const std::vector<double>& vec){
    std::cout <<  name << " = \n";
    std::for_each(vec.begin(), vec.end(), [](const double val){std::cout << val << " ";});
    std::cout << "\n";
  }


};


#endif//MYBOW_SRC_EXPERIMENTAL_STATS_H_
