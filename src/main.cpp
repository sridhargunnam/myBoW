#include<iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/core.hpp> // FileStorage

#include "experimental/Stats.h"


#include <iterator>
#include <numeric>


typedef std::vector<std::string> stringvec;
struct path_leaf_string
{
  std::string operator()(const std::filesystem::directory_entry& entry) const
  {
    return entry.path().string();
  }
};

void ReadDirectory(const std::string& dir_path, stringvec& v,  std::string& data_set_name)
{
  std::filesystem::path p(dir_path);
  // there is no reverse iteartor for filesystem::path
  for(auto i:p){
    data_set_name = i;
  }
  std::filesystem::directory_iterator start(p);
  std::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(v), path_leaf_string());
}

void ShowImage(std::string title, const cv::Mat& input_image){
  cv::namedWindow(title, cv::WINDOW_NORMAL);
  cv::imshow(title, input_image);
  cv::waitKey(0);
}

// TODO get rid of this
const std::vector<cv::KeyPoint> ComputeSiftKeypoints(const cv::Mat& input_image) {
  auto detector = cv::SIFT::create(100);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detect(input_image, keypoints);
  detector->compute(input_image, keypoints, descriptors);
  std::cout << "Descriptors = " << descriptors << "\n";
  return keypoints;
}

void AddDescriptorsToList(std::vector<cv::KeyPoint>& list_of_keypoints, std::vector<cv::Mat> &list_of_descriptors, std::string input_image, const int num_of_descriptor= 100){
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

int Plot(){
  {
    cv::Mat data_x( 1, 51, CV_64F );
    cv::Mat data_y( 1, 51, CV_64F );

    for ( int i = 0; i < data_x.cols; i++ )
    {
      double x = ( i - data_x.cols / 2 );
      data_x.at<double>( 0, i ) = x;
      data_y.at<double>( 0, i ) = x * x * x;
    }

    std::cout << "data_x : " << data_x << std::endl;
    std::cout << "data_y : " << data_y << std::endl;

    cv::Mat plot_result;

    cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create( data_x, data_y );
    plot->render(plot_result);

    imshow( "The plot rendered with default visualization options", plot_result );

    plot->setShowText( false );
    plot->setShowGrid( false );
    plot->setPlotBackgroundColor( cv::Scalar( 255, 200, 200 ) );
    plot->setPlotLineColor( cv::Scalar( 255, 0, 0 ) );
    plot->setPlotLineWidth( 2 );
    plot->setInvertOrientation( true );
    plot->render( plot_result );

    imshow( "The plot rendered with some of custom visualization options", plot_result );
    cv::waitKey();

    return 0;
  }
}

std::vector<cv::Scalar> PopulateColors(int N=1){
  cv::RNG rng(12345);
  std::vector<cv::Scalar> colors;
  for (int i = 0; i < N; i++) {
    cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    colors.push_back(color);
  }
  return colors;
}

inline void DrawKeyPoints(cv::Mat image, const std::vector<cv::KeyPoint>&  list_of_keypoints,
                      const std::size_t& keypoint_index, cv::Scalar& color ){
  cv::drawKeypoints(image, {list_of_keypoints[keypoint_index]}, image, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

template<typename T>
void SaveToFile(std::string filename, T item){
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "item" << item;
}

template<typename T>
void LoadFromFile(std::string filename, T& item){
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  fs["item"] >> item;
}

void ProcessInputs( const std::string& data_set_dir,  const bool read_images,
                    std::vector<cv::Mat>& list_of_descriptors, std::vector<cv::KeyPoint>& list_of_keypoints,
                    stringvec& file_list)
{

  std::string data_set_name;
  ReadDirectory(data_set_dir, file_list, data_set_name);
  // This file should be in bin directory
  std::cout << "dataset name = " << data_set_name << "\n";
  std::string descriptors_and_key_points_file = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/save_dir/"
                                                + data_set_name + ".yml.gz";

  if(read_images) {
    for (auto f : file_list) {
      AddDescriptorsToList(list_of_keypoints, list_of_descriptors, f);
    }
    // Save descriptors
    {
      cv::FileStorage fs(descriptors_and_key_points_file, cv::FileStorage::WRITE);
      std::cout << "writing keypoints and descriptors\n";
      fs << "list_of_descriptors" << list_of_descriptors;
      fs << "list_of_keypoints" << list_of_keypoints;
    }
  }

  else {
    //read
    {
      cv::FileStorage fs(descriptors_and_key_points_file, cv::FileStorage::READ);
      if (!fs.isOpened()) {
        std::cerr << "failed to open " << descriptors_and_key_points_file << std::endl;
        exit(0);
      }
      std::cout << "reading Descriptors and keypoints" << std::endl;
      fs["list_of_descriptors"] >> list_of_descriptors;
      fs["list_of_keypoints"] >> list_of_keypoints;
    }
  }
}


void WriteDebugImage(const std::string& filename, cv::Mat& image){
  std::filesystem::path p(filename);
  // there is no reverse iterator for filesystem::path
  std::string temp{};
  std::string temp_prev;
  for(auto i:p){
    temp_prev = temp;
    temp = i;
  }
  std::string debug_image_file_path = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/debug_images/" + temp_prev + "/";
  debug_image_file_path += temp;
  cv::imwrite(debug_image_file_path,image);
}

template<typename T>
void CheckCurrentHistogramCount(std::vector<T>& current_image_hist, const int& total_desc_in_image){
  for (auto h : current_image_hist) {
  std::cout << h << ", ";
  }
  std::cout << "\n";

  auto sum = std::accumulate(current_image_hist.begin(), current_image_hist.end(), static_cast<T>(0));
  if(sum == total_desc_in_image){
  std::cout << "Matching descriptors    " << sum << "\n";
  } else{
  std::cout << "Mismatching descriptors " << sum << "\n";
  }
}

int main(int ac, char** av){
  // read_images_from_disk  enable_debug_write_image
  // false        true
  cv::CommandLineParser parser(ac, av,"{@read_images||}{@enable_debug_write_images||}{@use_saved_labels_centers_hists||}");
  auto read_images                    = parser.get<bool>("@read_images");
  bool enable_debug_write_images      = parser.get<bool>("@enable_debug_write_images");
  bool use_saved_labels_centers_hists = parser.get<bool>( "@use_saved_labels_centers_hists");

  stringvec file_list;
  std::vector<cv::Mat> list_of_descriptors;
  std::vector<cv::KeyPoint> list_of_keypoints;

  std::vector<std::vector<int>> all_hists;
  std::vector<std::vector<double>> all_hists_normalized;
  const int kCentroids = 100;
  cv::Mat labels;
  cv::Mat centers;
  std::string fname_centers = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/save_dir/centers.yml.gz";
  std::string fname_labels = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/save_dir/labels.yml.gz";
  std::string fname_hists = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/save_dir/hists.yml.gz";
  std::string fname_norm_hists = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/save_dir/norm_hists.yml.gz";

  if(!use_saved_labels_centers_hists){
    /*
   * Reading the training data set and finding the descriptors and keypoints and writing them to file.
   * If the Keypoints and descriptors already exist, and if the read_option is "read_images" is false, then will read the
   * descriptors and keypoints from the previously generated results.
   */
    std::string data_set_dir = "/mnt/data/ws/Evaluation/cv/unio-bonn-cpp/Bag_of_Visual_Words/myRoom/training";

    ProcessInputs(data_set_dir, read_images, list_of_descriptors, list_of_keypoints, file_list);

    cv::Mat all_descriptors;
    for (auto i : list_of_descriptors) {
      all_descriptors.push_back(i);
    }

    all_descriptors.convertTo(all_descriptors, CV_32F);

    kmeans(all_descriptors, kCentroids, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 1), 3, cv::KMEANS_PP_CENTERS, centers);
    std::vector<int> flattened_labels(labels.begin<int>(), labels.end<int>());


    auto colors = PopulateColors(kCentroids);

    auto label_iterator_start = flattened_labels.begin();
    auto label_iterator_end = flattened_labels.begin();
    auto keypoint_index = 0ul;
    auto current_image_desc_iter = list_of_descriptors.begin();

    for (auto file : file_list) {
      std::cout << "Generating histograms and debug images for image " << file << "\n";
      auto total_desc_in_image = (*current_image_desc_iter).rows;
      std::advance(current_image_desc_iter, 1);
      std::advance(label_iterator_end, total_desc_in_image);

      std::vector<double> current_image_normalized_hist;
      std::vector<int> current_image_hist(kCentroids);
      cv::Mat image = cv::imread(file);

      while (std::distance(label_iterator_start, label_iterator_end) > 0) {
        //if (keypoint_index == static_cast<unsigned long>(total_desc_in_image)) {break;}
        auto idx = static_cast<std::size_t>(*label_iterator_start);
        std::advance(label_iterator_start, 1);
        ++current_image_hist[idx];
        if (enable_debug_write_images) { DrawKeyPoints(image, list_of_keypoints, keypoint_index, colors[idx]); }
        ++keypoint_index;
      }
      if (enable_debug_write_images) { WriteDebugImage(file, image); }

      for (auto bin : current_image_hist) {
        current_image_normalized_hist.push_back(static_cast<double>(bin) / total_desc_in_image);
      }
      all_hists_normalized.push_back(current_image_normalized_hist);
      all_hists.push_back(current_image_hist);
      //CheckCurrentHistogramCount<int>(current_image_hist, total_desc_in_image);
      CheckCurrentHistogramCount<double>(current_image_normalized_hist, total_desc_in_image);
    }

    // TFIDF
    for (auto hist : all_hists_normalized) {
      //std::sort(hist.begin(), hist.end());
      auto mean = std::accumulate(hist.begin(), hist.end(), 0.0) / static_cast<double>(hist.size());
      std::cout << "Mean:" << mean << std::endl;
      CheckCurrentHistogramCount<double>(hist, 1);
    }

    /*
      (n*(n-1))/(2) diffs
      compare to threshold t = 0.001
      if(diff lesser than threshold]
         count plus 1
         print count
      */

    // Save all the centroids and histograms
      SaveToFile(fname_centers, centers );
      SaveToFile(fname_labels, labels );
      SaveToFile(fname_hists, all_hists );
      SaveToFile(fname_norm_hists, all_hists_normalized);



    } else {
    // Save all the centroids and histograms
//      LoadFromFile(fname_labels, labels );
//      LoadFromFile(fname_hists, all_hists );
//      LoadFromFile(fname_centers, centers );
      LoadFromFile(fname_norm_hists, all_hists_normalized);
    }

    Stats stats{all_hists_normalized};
    stats.ComputeStats();
    return 0;
    /*
  std::vector<int> countDiffsAboveThresholdVec;
  for(int c=0; c<kCentroids; c++){
    std::vector<double> diffs;
    //diffs.reserve(kCentroids*(kCentroids-1)/2);
    std::vector<double> items;
    for(auto h:all_hists_normalized){
      items.push_back(h[c]);
    }

    for(int j=0; j<(15-1); j++){
      for(int k=(j+1); k < 15; k++){
        diffs.push_back(std::abs(items[j] - items[k]));
      }
    }

    int countDiffsAboveThreshold = 0;
    double threshold = 0.01;
    countDiffsAboveThreshold = std::count_if(diffs.begin(), diffs.end(), [&](double d){return d>threshold;});
    countDiffsAboveThresholdVec.push_back(countDiffsAboveThreshold);
    std::cout << "Total N(number of images) = " << 15 << " ni(number of occurrences) = " <<  countDiffsAboveThreshold << std::endl;

  }

  std::cout << "countDiffsAboveThresholdVec \n";
  std::sort(countDiffsAboveThresholdVec.begin(), countDiffsAboveThresholdVec.end());
  for(auto i:countDiffsAboveThresholdVec){
    std::cout << i << ", ";
  }
  std::cout << "\n";

    return 0;
     */
}

