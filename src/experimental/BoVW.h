//
// Created by sgunnam on 9/17/20.
//

#ifndef MYBOW_BOVW_H
#define MYBOW_BOVW_H
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <fstream>
#include <stdio.h>
#include <stdarg.h>
#include<unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/core.hpp>// FileStorage

#include "Stats.h"

#define NAME_OF(variable) #variable

struct BoVWParams
{
  std::size_t kCentroids = 30;
  std::size_t number_of_desc_per_image = 150;
  std::string dataset_dir = "/home/sgunnam/wsp/CLionProjects/myBoW/data";
  std::string dataset_name = "/smallImageMinimal"; // "/smallImagedataset"; //    //"/myRoom";
  std::string training_images_folder = "/training";
  std::string test_images_folder = "/test_images";
  std::string debug_images_folder = "/debug_images";
  std::string save_data_folder = "/save_dir";
  bool parallel_algorithms = true;
  bool include_TFIDF = true;
} ;

class BoVW
{
private:
  BoVWParams boVwParams_;
  bool read_images = false;
  bool enable_debug_write_images = false;
  bool use_saved_labels_centers_hists = true;
  std::vector<std::string> training_file_list;
  std::unordered_set<std::string> training_images_to_skip;
  std::vector<std::string> testing_file_list;
  std::vector<std::vector<cv::KeyPoint>> list_of_keypoints_seperately;
  std::vector<cv::Mat> list_of_descriptors_seperately;
  std::vector<cv::Mat> list_of_descriptors;
  std::vector<cv::KeyPoint> list_of_keypoints;
  std::vector<std::vector<int>> all_hists;
  std::vector<std::vector<double>> all_hists_normalized;
  cv::Mat labels;
  cv::Mat centers;
  std::vector<double> hist_tfidf_ni;
  std::vector<std::vector<double>> costMatrix;
  std::string save_dir_path;
public:
  BoVW(BoVWParams& boVwParams);

private:
  void ProcessInputs();
  void CreateDictionary();
  void testBoVW();

  void ProcessInputsParallel();
  static std::vector<std::size_t> CreateTestImageHistogram(
    const cv::Mat& descriptors_test,
    const cv::Mat& centers,
    std::vector<std::size_t>& label_test);


    // helper function, can be moved outside the class
  void ReadDirectory(std::string &data_set_path,
                      std::vector<std::string> &training_file_list);
  static std::vector<cv::Scalar> PopulateColors(std::size_t);

  template<typename T>
  void CheckCurrentHistogramCount(std::vector<T>& current_image_hist,
                      const int& total_desc_in_image);

  void CreateFile(const std::string &file_name);
  template<typename T>
  void SaveToFile(const std::string& filename, T item);
  template<typename T>
  void LoadFromFile(const std::string& filename, T& item);

  void ShowManyImagesForBoVW(const std::string& title, const std::string& test_image, const std::vector<std::string>& matching_images) ;

  double ComputeAccuracy(const std::string& test_file, const std::vector<std::string>& matching_files);
  };


#endif//MYBOW_BOVW_H
