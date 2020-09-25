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

struct path_leaf_string
{
  std::string operator()(const std::filesystem::directory_entry& entry) const
  {
    return entry.path().string();
  }
};

void ReadDirectory(const std::string& dir_path, std::vector<std::string>& v,  std::string& data_set_name)
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

//#include "experimental/showMultipleImages.h"
void AddDescriptorsToList(std::vector<cv::KeyPoint>& list_of_keypoints, std::vector<cv::Mat> &list_of_descriptors, std::string input_image, const int num_of_descriptor= 500){  cv::Mat im = cv::imread(input_image);
  auto detector = cv::SIFT::create(10); //num_of_descriptor);
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

template<typename T>
void Plot(const std::vector<std::string>& file_list, std::vector<std::vector<T>>& all_hists_normalized ){
  cv::namedWindow("Plot window", cv::WINDOW_KEEPRATIO);
  cv::namedWindow("Image window",  cv::WINDOW_KEEPRATIO);
    for (auto index=0; index < file_list.size(); ++index) {
      cv::Mat img = cv::imread(file_list[index], cv::IMREAD_COLOR);
      cv::waitKey(0);
      std::cout << "file_name " << file_list[index] << "\n";
      cv::imshow("Image window", img);
      std::vector<T> current_hist = all_hists_normalized[index];
      index++;
      auto len = static_cast<int>(current_hist.size());
      cv::Mat data_x(1, len, CV_64F);
      cv::Mat data_y(1, len, CV_64F);

      for (int i = 0; i < len; i++) {
        double x = static_cast<double>(i);
        data_x.at<double>(0, i) = x;
        data_y.at<double>(0, i) = current_hist[i];
      }

      std::cout << "data_x : " << data_x << std::endl;
      std::cout << "data_y : " << data_y << std::endl;

      cv::Mat plot_result;

      cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create(data_x, data_y);
      plot->render(plot_result);

      imshow("Plot window", plot_result);

      plot->setShowText(false);
      plot->setShowGrid(false);
      plot->setPlotBackgroundColor(cv::Scalar(255, 200, 200));
      plot->setPlotLineColor(cv::Scalar(255, 0, 0));
      plot->setPlotLineWidth(2);
      plot->setInvertOrientation(true);
      plot->render(plot_result);

      imshow("Plot window", plot_result);
      cv::waitKey();
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
                    std::vector<std::string>& file_list)
{

  std::string data_set_name;
  ReadDirectory(data_set_dir, file_list, data_set_name);
  // This file should be in bin directory
  std::cout << "dataset name = " << data_set_name << "\n";
  std::string descriptors_and_key_points_file = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/"
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
  }   else {
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
  std::string debug_image_file_path = "/home/sgunnam/wsp/CLionProjects/myBoW/data/debug_images/" + temp_prev + "/";
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

std::vector<int> CreateTestImageHistogram(const cv::Mat& descriptors_test, const cv::Mat& centers, std::vector<int>& label_test);

// Parameters that affect the result
/*
 * Dataset: images of objects only
 * Number of centroids - runtime effect
 * Number of descriptors from each image - runtime
 * Thresholding logic - engineering effort
 * Type of descriptors - SURF, SIFT
 */
int main(int ac, char** av){
  // read_images_from_disk  enable_debug_write_image
  // false        true
  // Arguments to reuse the saved desc data: false false true
  cv::CommandLineParser parser(ac, av,"{@read_images||}{@enable_debug_write_images||}{@use_saved_labels_centers_hists||}");
  bool read_images                    = parser.get<bool>("@read_images");
  bool enable_debug_write_images      = parser.get<bool>("@enable_debug_write_images");
  bool use_saved_labels_centers_hists = parser.get<bool>( "@use_saved_labels_centers_hists");

  std::vector<std::string> file_list;
  std::vector<cv::Mat> list_of_descriptors;
  std::vector<cv::KeyPoint> list_of_keypoints;

  std::vector<std::vector<int>> all_hists;
  std::vector<std::vector<double>> all_hists_normalized;
  const int kCentroids = 5;
  cv::Mat labels;
  cv::Mat centers;
  std::string fname_centers = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/centers.yml.gz";
  std::string fname_labels = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/labels.yml.gz";
  std::string fname_hists = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/hists.yml.gz";
  std::string fname_norm_hists = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/norm_hists.yml.gz";
  std::string fname_file_list = "/home/sgunnam/wsp/CLionProjects/myBoW/data/save_dir/file_list.yml.gz";

  if(!use_saved_labels_centers_hists){
    /*
   * Reading the training data set and finding the descriptors and keypoints and writing them to file.
   * If the Keypoints and descriptors already exist, and if the read_option is "read_images" is false, then will read the
   * descriptors and keypoints from the previously generated results.
   */
    std::string data_set_dir = "/home/sgunnam/wsp/CLionProjects/myBoW/data/myRoom/training/";

    ProcessInputs(data_set_dir,
      read_images,
      list_of_descriptors,
      list_of_keypoints,
      file_list);

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
      SaveToFile(fname_file_list, file_list);
    } else {
//     Save all the centroids and histograms
//      LoadFromFile(fname_labels, labels );
//      LoadFromFile(fname_hists, all_hists );
      LoadFromFile(fname_centers, centers );
      LoadFromFile(fname_norm_hists, all_hists_normalized);
      LoadFromFile(fname_file_list, file_list);
    }


    Stats<double> stats{all_hists_normalized, file_list};
    //Plot(file_list, all_hists_normalized);
    //auto all_variance = stats.getVariance();
    auto all_mean = stats.getMean();
    std::vector<double> hist_tfidf_ni(all_hists_normalized[0].size(),0);
    //TF-IDF
    for(int i=0; i<all_hists_normalized.size(); i++){
      for(auto j=0; j<all_hists_normalized[0].size(); j++){
        if(all_hists_normalized[i][j] <= all_mean[j]){
          hist_tfidf_ni[j] += 1;
        }
      }
    }
    PrintVecContainer("hist_tfidf_ni", hist_tfidf_ni);
    std::for_each(hist_tfidf_ni.begin(), hist_tfidf_ni.end(),
      [&](auto& first){
        first = std::log(all_hists_normalized.size()/first);
      });
      PrintVecContainer("hist_tfidf_ni after log operation", hist_tfidf_ni);
//    PrintVecContainer("IDF", hist_tfidf_ni);
//    PrintVecContainer("hist_tfidf_ni", hist_tfidf_ni, 7);

    for(auto& hi: all_hists_normalized){
      //std::cout << "----------------------------------------------------\n";
      //PrintVecContainer("hi_before", hi, 7);
      //PrintVecContainer("hist_tfidf_ni", hist_tfidf_ni, 7);
      std::transform(hi.begin(), hi.end(), hist_tfidf_ni.cbegin(), hi.begin(),
        [](auto& first, const auto second){ first = first*second; return first; });
      //PrintVecContainer("hi_after", hi, 7);
    }

  std::vector<std::vector<double>> costMatrix{all_hists_normalized.size(),
                                              std::vector<double>(all_hists_normalized.size())};
  auto costMatrixIter = costMatrix.begin();
  for(auto m=0; m<all_hists_normalized.size(); ++m){
    for(auto n=0; n< all_hists_normalized.size(); ++n){
      std::vector<double>    temp(all_hists_normalized[0].size(), 0);
      for(auto i=0; i<all_hists_normalized[0].size(); i++){
        temp[i] = std::pow((all_hists_normalized[m][i]-all_hists_normalized[n][i]), 2);
        //std::cout << "temp = " << temp[i] << "\n";
        //std::cout << "all_hists_normalized[m][i]" << all_hists_normalized[m][i] << "\n";
        //std::cout << "all_hists_normalized[n][i]" << all_hists_normalized[n][i] << "\n";
      }

      double accum =0;
      for(auto temp_item:temp){
        accum += temp_item;
      }
      costMatrix[m][n] = accum*100;
    }

    /*
    for(auto file_item:file_list){
      std::cout << file_item << "\n";
    }

    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    for(auto i:costMatrix){
      for(auto j:i){
        std::cout << j << ", ";
      }
      std::cout << "\n";
    }
    */
  }

  std::string test_image_filename = "/home/sgunnam/wsp/CLionProjects/myBoW/data/myRoom/training/IMG_20200823_125045.jpg";
  //"/home/sgunnam/wsp/CLionProjects/myBoW/data/colombia100Objs/testdata_coil100/obj6__0.png";
  cv::Mat im_test = cv::imread(test_image_filename);
  //showManyImages("an image in showMany images", 2, im_test, im_test);
  auto detector_test = cv::SIFT::create(10); //num_of_descriptor);
  std::vector<cv::KeyPoint> keypoints_test;
  cv::Mat descriptors_test;
  detector_test->detect(im_test, keypoints_test);
  detector_test->compute(im_test, keypoints_test, descriptors_test);
  std::vector<int> label_test;

  std::vector<int> test_hist =
      CreateTestImageHistogram(descriptors_test, centers, label_test);
  //Normalize
  std::vector<double> test_hist_norm(test_hist.size());
  std::copy(test_hist.begin(), test_hist.end(), test_hist_norm.begin());
  auto sum_test_hist = std::accumulate(test_hist.begin(), test_hist.end(), 0);
  std::cout << "Normalized Histogram = \n";
  std::for_each(test_hist_norm.begin(), test_hist_norm.end(), [&](auto &val){val = val/sum_test_hist; std::cout << val << ", "; });
  auto test_hist_norm_check_sum = std::accumulate(test_hist_norm.begin(), test_hist_norm.end(), 0.0);
  //assert(test_hist_norm_check_sum == 1);

  PrintVecContainer("Before test_hist_norm", test_hist_norm);
  std::transform(test_hist_norm.begin(), test_hist_norm.end(), hist_tfidf_ni.cbegin(), test_hist_norm.begin(),
                 [](auto& first, const auto second){ first = first*second; return first; });
  PrintVecContainer("After test_hist_norm", test_hist_norm);

  std::vector<double> ssd_vec(file_list.size(), 0.0);
  std::size_t file_index = 0;
  for(auto c:costMatrix){
    ssd_vec[file_index] =  std::inner_product(c.begin(), c.end(), test_hist_norm.begin(), 0.0,
                                                     std::plus<>(), [](auto&l, auto& r){ return std::pow((l-r),2);});
    std::cout << "ssd_vec[" << file_index << "] = " << ssd_vec[file_index] << std::endl;
    file_index++;
  }


  PrintVecContainer("ssd_vec : ", ssd_vec);
  auto ssd_vec_copy{ssd_vec};
  std::sort(ssd_vec_copy.begin(), ssd_vec_copy.end());

  std::vector<std::string> matching_files;
  std::size_t file_index2 = 0;
  for(auto s:ssd_vec){
    if(s <= ssd_vec_copy[2]){
      matching_files.push_back(file_list[file_index2]);
    }
    file_index2++;
  }
  //std::copy_n(file_list.begin(), 5, matching_files.begin());
//  for(auto i=0; i<matchCount; i++){
//    matching_files.at(i) = file_list.at(i);
//  }
  //ShowManyImagesForBoVW("BoVW image matching", test_image_filename, matching_files);
  return 0;
}

std::vector<int> CreateTestImageHistogram(const cv::Mat& descriptors_test, const cv::Mat& centers, std::vector<int>& label_test){
  int desc_count      = descriptors_test.rows;
  int centroids_count = centers.rows;
  label_test.resize(desc_count, INT_MAX);
  for(int i=0; i<desc_count; i++){
    auto euclidean_ssd = DBL_MAX;
    for(int j=0; j<centroids_count; j++){
      auto v1 = descriptors_test.row(i);
      auto v2 = centers.row(j);
      auto euclidean_dist = ((cv::abs(v1-v2)));
      cv::MatExpr euclidian_dist_sqr = euclidean_dist.mul(euclidean_dist);
      if (sum(euclidian_dist_sqr)[0] < euclidean_ssd) {
        euclidean_ssd = sum(euclidian_dist_sqr)[0];
        label_test[i] = j;
      }

    }
  }

  // generate histogram
  std::vector<int> test_hist(centers.size[0], 0);
  for(auto lab:label_test)
    ++test_hist.at(lab);
  return test_hist;
}