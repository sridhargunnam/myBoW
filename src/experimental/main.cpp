#include<iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>


#include "customFilter.h"
#include "fileReader.h"
#include "ImageHelper.h"

int main(){
  std::string data_set_dir = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/myRoom/training";
  stringvec file_list;
  read_directory(data_set_dir, file_list);

  std::vector<cv::Mat> list_of_descriptors;
  for(auto f:file_list){
    addDescriptorsToList(list_of_descriptors, f, 10);
    std::cout << f << "\n";
    break;
  }

  auto one_set_of_descriptors = list_of_descriptors[0];
  std::cout << "total = " << one_set_of_descriptors.total() << "\n";
  one_set_of_descriptors.convertTo(one_set_of_descriptors, CV_32F);
  cv::Mat labels;
  cv::Mat centers;
  cv::kmeans(one_set_of_descriptors, 10, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,
                                                              10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
  cv::BOWKMeansTrainer(10,cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,
                                                             10, 1.0), 3,cv::KMEANS_PP_CENTERS);
  std::cout << "\n";
  /*

// convert to float & reshape to a [3 x W*H] Mat
//  (so every pixel is on a row of it's own)
Mat data;
ocv.convertTo(data,CV_32F);
data = data.reshape(1,data.total());

// do kmeans
Mat labels, centers;
kmeans(data, 8, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
       KMEANS_PP_CENTERS, centers);

// reshape both to a single row of Vec3f pixels:
centers = centers.reshape(3,centers.rows);
data = data.reshape(3,data.rows);

// replace pixel values with their center value:
Vec3f *p = data.ptr<Vec3f>();
for (size_t i=0; i<data.rows; i++) {
   int center_id = labels.at<int>(i);
   p[i] = centers.at<Vec3f>(center_id);
}
*/

  return 0;
}












// TODO
// check out the tests for kmeans in opencv library, but give an attempt before checking it out
// https://github.com/opencv/opencv/blob/01a28db949e96e7c23cf618e5c83866e4f9b6f02/modules/ml/test/test_kmeans.cpp