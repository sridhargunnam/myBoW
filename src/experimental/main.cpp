#include<iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const cv::Mat readImage(std::string filename){
  return cv::imread(filename, cv::IMREAD_COLOR);
}

void showImage(std::string title, const cv::Mat& input_image){
  cv::namedWindow(title, cv::WINDOW_NORMAL);
  cv::imshow(title, input_image);
  cv::waitKey(0);
}


typedef std::vector<std::string> stringvec;
struct path_leaf_string
{
  std::string operator()(const std::filesystem::directory_entry& entry) const
  {
    return entry.path().string();
  }
};

void read_directory(const std::string& name, stringvec& v)
{
  std::filesystem::path p(name);
  std::filesystem::directory_iterator start(p);
  std::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(v), path_leaf_string());
}

int customFilter(const std::string filename){
  // Declare variables
  cv::Mat src, dst;
  const char* window_name = "filter2D Demo";;
  // Loads an image
  src = imread( cv::samples::findFile( filename ), cv::IMREAD_COLOR ); // Load an image
  if( src.empty() )
  {
    printf(" Error opening image\n");
    return EXIT_FAILURE;
  }
  // Initialize arguments for the filter
  cv::Mat kernel;
  cv::Point anchor = cv::Point( -1, -1 );
  double delta = 0;
  int ddepth = -1;
  // Loop - Will filter the image with different kernel sizes each 0.5 seconds
  int ind = 0;
  for(;;)
  {
    int kernel_size;
    // Update kernel size for a normalized box filter
    kernel_size = 3 + 2*( ind%5 );
    kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (kernel_size*kernel_size);
    // Apply filter
    cv::filter2D(src, dst, ddepth , kernel, anchor, delta, cv::BORDER_DEFAULT );
    imshow( window_name, dst );
    char c = static_cast<char>(cv::waitKey(500));
    // Press 'ESC' to exit the program
    if( c == 27 )
    { break; }
    ind++;
  }
  return EXIT_SUCCESS;
}
int main(){
  std::string dir_path{"/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/myRoom"};
  stringvec v;
  read_directory(dir_path, v);
  for(auto i:v){
    showImage(i, readImage(i));
    customFilter(i);
    break;
  }

  return 0;
}
