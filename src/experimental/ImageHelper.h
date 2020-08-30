//
// Created by srigun on 8/25/20.
//

#ifndef MYBOW_SRC_EXPERIMENTAL_IMAGEHELPER_H_
#define MYBOW_SRC_EXPERIMENTAL_IMAGEHELPER_H_
#include<iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator>

const cv::Mat readImage(std::string filename){
  return cv::imread(filename, cv::IMREAD_COLOR);
}

void ShowImage(std::string title, const cv::Mat& input_image){
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
  cv::imshow(title, input_image);
  cv::waitKey(0);
}

// TODO get rid of this
const std::vector<cv::KeyPoint> ComputeSiftKeypoints(const cv::Mat& input_image) {
  //CV_WRAP static Ptr<SIFT> create(int nfeatures = 0, int nOctaveLayers = 3,
//                                  double contrastThreshold = 0.04, double edgeThreshold = 10,
//                                  double sigma = 1.6);
  auto detector = cv::SIFT::create(100);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detect(input_image, keypoints);
  detector->compute(input_image, keypoints, descriptors);
  std::cout << "Descriptors = " << descriptors << "\n";
  return keypoints;
/*
  std::vector<cv::KeyPoint> keypoints {ComputeSiftKeypoints(readImage(file_list[0]))};
  cv::Mat output;
  Drawpoints(readImage(file_list[0]), keypoints,output);
  ShowImage("output", output);
  */
}

void addDescriptorsToList(std::vector<cv::Mat> &list_of_descriptors, std::string input_image, const int num_of_descriptor=100){
  cv::Mat im = cv::imread(input_image);
  auto detector = cv::SIFT::create(num_of_descriptor);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detect(im, keypoints);
  detector->compute(im, keypoints, descriptors);
  //std::cout << "Descriptors = " << descriptors << "\n";
  list_of_descriptors.push_back(descriptors);
}

void drawpoints(const cv::Mat& input_image, std::vector<cv::KeyPoint> keypoints, cv::Mat& output_image){
  cv::drawKeypoints(input_image, keypoints, output_image);
  //cv::imwrite("sift_result.jpg",output_image);
}

#endif//MYBOW_SRC_EXPERIMENTAL_IMAGEHELPER_H_
