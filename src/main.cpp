#include<iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator>


typedef std::vector<std::string> stringvec;
struct path_leaf_string
{
  std::string operator()(const std::filesystem::directory_entry& entry) const
  {
    return entry.path().string();
  }
};

void read_directory(const std::string& dir_path, stringvec& v)
{
  std::filesystem::path p(dir_path);
  std::filesystem::directory_iterator start(p);
  std::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(v), path_leaf_string());
}


const cv::Mat readImage(std::string filename){
  return cv::imread(filename, cv::IMREAD_COLOR);
}

void showImage(std::string title, const cv::Mat& input_image){
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
  cv::imshow(title, input_image);
  cv::waitKey(0);
}

// TODO get rid of this
const std::vector<cv::KeyPoint> computeSIFTKeypoints(const cv::Mat& input_image) {
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
//  CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
//                              TermCriteria criteria, int attempts,
//                              int flags, OutputArray centers = noArray() );
  kmeans(one_set_of_descriptors, 10, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,
                                               10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

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

// back to 2d, and uchar:
ocv = data.reshape(3, ocv.rows);
ocv.convertTo(ocv, CV_8U);
   */


//  std::vector<cv::KeyPoint> keypoints {computeSIFTKeypoints(readImage(file_list[0]))};
//  cv::Mat output;
//  drawpoints(readImage(file_list[0]), keypoints,output);
//  showImage("output", output);
  return 0;
}