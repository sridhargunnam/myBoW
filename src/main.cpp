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
  cv::namedWindow(title, cv::WINDOW_NORMAL);
  cv::imshow(title, input_image);
  cv::waitKey(0);
}

// TODO get rid of this
const std::vector<cv::KeyPoint> computeSIFTKeypoints(const cv::Mat& input_image) {
  auto detector = cv::SIFT::create(100);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detect(input_image, keypoints);
  detector->compute(input_image, keypoints, descriptors);
  std::cout << "Descriptors = " << descriptors << "\n";
  return keypoints;
}

void addDescriptorsToList(std::vector<cv::KeyPoint>& list_of_keypoints, std::vector<cv::Mat> &list_of_descriptors, std::string input_image, const int num_of_descriptor=100){
  cv::Mat im = cv::imread(input_image);
  std::cout << num_of_descriptor << std::endl;
  auto detector = cv::SIFT::create(); //num_of_descriptor);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detect(im, keypoints);
  detector->compute(im, keypoints, descriptors);
  //std::cout << "Descriptors = " << descriptors << "\n";
  list_of_descriptors.push_back(descriptors);
  list_of_keypoints.insert(list_of_keypoints.end(), keypoints.begin(), keypoints.end());
}

void drawpoints(cv::Mat& input_image, std::vector<cv::KeyPoint> keypoints, cv::Mat& output_image,  //{
                const cv::Scalar color = cv::Scalar::all(-1) ) {
    cv::drawKeypoints(input_image, keypoints, output_image, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //cv::imwrite("sift_result.jpg",output_image);
}
//
//void drawHist(const std::vector<int>& data, cv::Mat3b& dst, std::size_t binSize = 1, std::size_t height = 0)
//{
//  auto max_value = *(std::max(data.begin(), data.end()));
//  std::size_t rows = 0;
//  std::size_t cols = 0;
//  if (height == 0) {
//    rows = static_cast<size_t>(max_value + 10);
//  } else {
//    rows = (std::max(static_cast<size_t>(max_value + 10), height));
//  }
//
//  cols = data.size() * binSize;
//
//  dst = cv::Mat(rows, cols, cv::Vec3b(0,0,0));
//
//  for (int i = 0; i < data.size(); ++i)
//  {
//    int h = rows - data[i];
//    rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), (i%2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255), CV_FILLED);
//  }
//
//}

int main(){
  std::string data_set_dir = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/myRoom/training";  // /home/srigun/Pictures/sift/star" ; //
  stringvec file_list;
  read_directory(data_set_dir, file_list);

  std::vector<cv::Mat> list_of_descriptors;
  std::vector<cv::KeyPoint> list_of_keypoints;
  for(auto f:file_list){
    addDescriptorsToList(list_of_keypoints,list_of_descriptors, f);
    std::cout << f << "\n";
    break;
  }

  cv::Mat all_descriptors;
  for(auto i:list_of_descriptors){
    all_descriptors.push_back(i);
    //std::cout << i << "\n";
    break;
  }

  std::cout << "total = " << all_descriptors.total() << "\n";
  all_descriptors.convertTo(all_descriptors, CV_32F);
  cv::Mat labels;
  cv::Mat centers;
  const int kCentroids = 6;
  kmeans(all_descriptors, kCentroids, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,
                                               10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
  std::cout << "\n";
  std::vector<int> flattened_labels(labels.begin<int>(), labels.end<int>());

  cv::Mat input_image = cv::imread(file_list[0]);
  int counter = 0;

//  cv::Mat output_image;
//  drawpoints(input_image, list_of_keypoints, output_image, cv::Scalar( 255, 10, 10 ));
//  //showImage("viola", output_image);
//  cv::imwrite("all_keys.jpg",output_image);
//  return 0;

  std::vector<int> hist(kCentroids);
  for(auto l:flattened_labels) {
    switch (l) {
      case 0: {
        drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 255, 10, 10 ));
        ++counter;
        ++hist[static_cast<unsigned long>(l)];
      }
        break;
      case 1:{
        drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 10, 255, 10 ));
        ++counter;
        ++hist[static_cast<unsigned long>(l)];
      }
        break;
      case 2: {
        drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 10, 10, 255 ));
        ++counter;
        ++hist[static_cast<unsigned long>(l)];
      }
        break;
    case 3: {
      drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 255, 255, 10 ));
      ++counter;
      ++hist[static_cast<unsigned long>(l)];
    }
        break;
    case 4: {
      drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 255, 10, 255 ));
      ++counter;
      ++hist[static_cast<unsigned long>(l)];
    }
      break;
    case 5: {
      drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 10, 255, 255 ));
      ++counter;
      ++hist[static_cast<unsigned long>(l)];
    }
      break;
    default: {
      std::cout << "I am in default. Shouldn't be here at label = " << l << "\n";
      drawpoints(input_image, { list_of_keypoints[static_cast<unsigned long>(counter)] }, input_image, cv::Scalar( 1, 1, 1 ));
      ++counter;
      ++hist[static_cast<unsigned long>(l)];
    }
      break;
    }
  }

  std::cout << "histogram here\n";
  for(auto h:hist){
    std::cout << h << ", " ;
  }
  std::cout << "\n";

//  showImage("viola", input_image);
//    cv::imwrite("all_keys.jpg",input_image);

  return 0;
}