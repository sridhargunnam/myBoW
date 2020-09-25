//
// Created by sgunnam on 9/17/20.
//

#include "BoVW.h"

namespace fs = std::filesystem;

BoVW::BoVW(BoVWParams& boVwParams) : boVwParams_(boVwParams)
{
  ProcessInputs();
  CreateDictionary();
  testBoVW();
}

struct path_leaf_string
{
  std::string operator()(const std::filesystem::directory_entry& entry) const
  {
    return entry.path().string();
  }
};

void BoVW::ReadDirectory(std::string &data_set_path,
                         std::vector<std::string> &file_list)
{
  std::filesystem::path p(data_set_path);
  std::string temp_data_set_name;
  for(auto i:p){
    temp_data_set_name = i;
  }
  //boVwParams_.dataset_name = "/" + temp_data_set_name;
  std::filesystem::directory_iterator start(p);
  std::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(file_list), path_leaf_string());
  std::cout << "\n";
}

void BoVW::ProcessInputs()
{
  auto data_set_path = boVwParams_.dataset_dir + boVwParams_.dataset_name + boVwParams_.training_images_folder;
  ReadDirectory(data_set_path, training_file_list);
  assert(!training_file_list.empty());
  std::string save_dir_path = boVwParams_.dataset_dir +
                              boVwParams_.dataset_name +
                              boVwParams_.save_data_folder
                              + "/";

  std::string list_of_descriptors_filename = save_dir_path
                                            + NAME_OF(list_of_descriptors) + ".yml.gz";
  std::string list_of_keypoints_filename = save_dir_path
                                             + NAME_OF(list_of_keypoints) + ".yml.gz";

  // TODO: Make reading files and adding descriptors multi threaded
  // Preallocate the memory desctiptors, and keypoints , so that there is no race condition
  if(read_images) {
    for (const auto& input_image : training_file_list) {
      cv::Mat im = cv::imread(input_image);
      auto detector = cv::SIFT::create(static_cast<int>(boVwParams_.number_of_desc_per_image));
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      detector->detect(im, keypoints);
      detector->compute(im, keypoints, descriptors);
      list_of_descriptors.push_back(descriptors);
      list_of_keypoints.insert(list_of_keypoints.end(), keypoints.begin(), keypoints.end());
    }
    // Save descriptors to file
    {
      // Create the file, as cv::FileStorage doesn't create one
      CreateFile(list_of_descriptors_filename);
      CreateFile(list_of_keypoints_filename);
      cv::FileStorage fs_desc(list_of_descriptors_filename, cv::FileStorage::WRITE);
      cv::FileStorage fs_keys(list_of_keypoints_filename, cv::FileStorage::WRITE);
      fs_desc << "list_of_descriptors" << list_of_descriptors;
      fs_keys << "list_of_keypoints" << list_of_keypoints;
    }
  }   else {
    //read from file if it already exist
    {
      if(!fs::exists(list_of_descriptors_filename) || !fs::exists(list_of_keypoints_filename)){
        std::cout << list_of_descriptors_filename <<
                    "or " <<  list_of_keypoints_filename << "does not exist\n";
        exit(0);
      }

      cv::FileStorage fs_desc(list_of_descriptors_filename, cv::FileStorage::READ);
      cv::FileStorage fs_keys(list_of_keypoints_filename, cv::FileStorage::READ);
      if (!fs_desc.isOpened() || !fs_keys.isOpened()) {
        std::cerr << "failed to open " <<  list_of_descriptors_filename << "or "
                  << list_of_keypoints_filename << std::endl;
        exit(0);
      }
      fs_desc["list_of_descriptors"] >> list_of_descriptors;
      fs_keys["list_of_keypoints"] >> list_of_keypoints;
    }
  }
}

void BoVW::CreateDictionary()
{
  if (!use_saved_labels_centers_hists) {
    //TODO try to remove the need for container to store all decriptors
    cv::Mat all_descriptors;
    for (const auto& i : list_of_descriptors) {
      all_descriptors.push_back(i);
    }

    all_descriptors.convertTo(all_descriptors, CV_32F);

    kmeans(all_descriptors,
      static_cast<int>(boVwParams_.kCentroids),
      labels,
      cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 1),
      3,
      cv::KMEANS_PP_CENTERS,
      centers);

    std::vector<int> flattened_labels(labels.begin<int>(), labels.end<int>());


    auto colors = PopulateColors(boVwParams_.kCentroids);

    auto label_iterator_start = flattened_labels.begin();
    auto label_iterator_end = flattened_labels.begin();
    auto keypoint_index = 0ul;
    auto current_image_desc_iter = list_of_descriptors.begin();


    for (const auto &file : training_file_list) {
      std::cout << "Generating histograms and debug images for image " << file << "\n";
      auto total_desc_in_image = (*current_image_desc_iter).rows;
      std::advance(current_image_desc_iter, 1);
      std::advance(label_iterator_end, total_desc_in_image);

      std::vector<double> current_image_normalized_hist;
      std::vector<int> current_image_hist(boVwParams_.kCentroids);
      cv::Mat image = cv::imread(file);

      while (std::distance(label_iterator_start, label_iterator_end) > 0) {
        //if (keypoint_index == static_cast<unsigned long>(total_desc_in_image)) {break;}
        auto idx = static_cast<std::size_t>(*label_iterator_start);
        std::advance(label_iterator_start, 1);
        ++current_image_hist[idx];
        if (enable_debug_write_images) {
          cv::drawKeypoints(image,
            { list_of_keypoints[keypoint_index] },
            image,
            colors[idx],
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
        ++keypoint_index;
      }
      if (enable_debug_write_images) {
        std::filesystem::path p(file);
        // TODO needs to be tested if debug image paths are correct
        std::string debug_images_dir_path = boVwParams_.dataset_dir +
                                            boVwParams_.dataset_name +
                                            boVwParams_.debug_images_folder
                                            + "/";
        std::string debug_image_file_path = debug_images_dir_path + p.filename().string();
        cv::imwrite(debug_image_file_path, image);
      }

      for (auto bin : current_image_hist) {
        current_image_normalized_hist.push_back(static_cast<double>(bin) / total_desc_in_image);
      }
      all_hists_normalized.push_back(current_image_normalized_hist);
      all_hists.push_back(current_image_hist);
      CheckCurrentHistogramCount<double>(current_image_normalized_hist, total_desc_in_image);
    }

    // TFIDF
    for (auto hist : all_hists_normalized) {
      //std::sort(hist.begin(), hist.end());
      auto mean = std::accumulate(hist.begin(), hist.end(), 0.0) / static_cast<double>(hist.size());
      std::cout << "Mean:" << mean << std::endl;
      CheckCurrentHistogramCount<double>(hist, 1);
    }

    std::string  save_data_path = boVwParams_.dataset_dir +
                                        boVwParams_.dataset_name +
                                        boVwParams_.save_data_folder
                                        + "/";
    // Save all the centroids and histograms
    SaveToFile(save_data_path + NAME_OF(centers) + ".yml.gz", centers);
    SaveToFile(save_data_path + NAME_OF(labels) + ".yml.gz", labels);
    SaveToFile(save_data_path + NAME_OF(all_hists) + ".yml.gz", all_hists);
    SaveToFile(save_data_path + NAME_OF(all_hists_normalized) + ".yml.gz", all_hists_normalized);
    SaveToFile(save_data_path + NAME_OF(training_file_list) + ".yml.gz", training_file_list);
  } else {
    std::string  save_data_path = boVwParams_.dataset_dir +
                                  boVwParams_.dataset_name +
                                  boVwParams_.save_data_folder
                                  + "/";
    LoadFromFile(save_data_path + NAME_OF(centers) + ".yml.gz", centers);
    LoadFromFile(save_data_path + NAME_OF(labels) + ".yml.gz", labels );
    LoadFromFile(save_data_path + NAME_OF(all_hists) + ".yml.gz", all_hists );
    LoadFromFile(save_data_path + NAME_OF(all_hists_normalized) + ".yml.gz", all_hists_normalized);
    LoadFromFile(save_data_path + NAME_OF(training_file_list) + ".yml.gz", training_file_list);
  }

  Stats<double> stats{ all_hists_normalized, training_file_list };
  auto all_mean = stats.getMean();
  hist_tfidf_ni.resize(all_hists_normalized[0].size(), 0);
  //TF-IDF
  for (auto & hist : all_hists_normalized) {
    for (auto j = 0ul; j < all_hists_normalized[0].size(); j++) {
      if (hist[j] <= all_mean[j]) {
        hist_tfidf_ni[j] += 1;
      }
    }
  }

  PrintVecContainer("hist_tfidf_ni", hist_tfidf_ni);
  std::for_each(hist_tfidf_ni.begin(), hist_tfidf_ni.end(),
    [&](auto &first) {
      first = std::log( static_cast<double>(all_hists_normalized.size()) / first);
    });
  PrintVecContainer("hist_tfidf_ni after log operation", hist_tfidf_ni);

  for (auto &hi : all_hists_normalized) {
    //std::cout << "----------------------------------------------------\n";
    //PrintVecContainer("hi_before", hi, 7);
    //PrintVecContainer("hist_tfidf_ni", hist_tfidf_ni, 7);
    std::transform(hi.begin(), hi.end(), hist_tfidf_ni.cbegin(), hi.begin(),
      [](auto &first, const auto second) { first = first*second; return first; });
    PrintVecContainer("hi_after", hi, 7);
  }

  costMatrix.resize( all_hists_normalized.size(),
    std::vector<double>(all_hists_normalized.size()) );
  //auto costMatrixIter = costMatrix.begin();
  for (std::size_t m = 0; m < all_hists_normalized.size(); ++m) {
    for (std::size_t  n = 0; n < all_hists_normalized.size(); ++n) {
      std::vector<double> temp(all_hists_normalized[0].size(), 0);
      for (std::size_t  i = 0; i < all_hists_normalized[0].size(); i++) {
        temp[i] = std::pow((all_hists_normalized[m][i] - all_hists_normalized[n][i]), 2);
      }

      double accum = 0;
      for (auto temp_item : temp) {
        accum += temp_item;
      }
      costMatrix[m][n] = accum * 100;
    }
  }

  for (auto &hi : all_hists_normalized) {
    PrintVecContainer("hi_after", hi, 7);
  }
}


std::vector<cv::Scalar> BoVW::PopulateColors(std::size_t N=1){
  cv::RNG rng(12345);
  std::vector<cv::Scalar> colors;
  for (std::size_t i = 0; i < N; i++) {
    cv::Scalar color = cv::Scalar(
      rng.uniform(0,255),
      rng.uniform(0, 255),
      rng.uniform(0, 255));
    colors.push_back(color);
  }
  return colors;
}

template<typename T>
void BoVW::CheckCurrentHistogramCount(std::vector<T>& current_image_hist,
  const int& total_desc_in_image){
  for (auto h : current_image_hist) {
    std::cout << h << ", ";
  }
  std::cout << "\n";

  auto sum = std::accumulate(current_image_hist.begin(),
                              current_image_hist.end(), static_cast<T>(0));
  if(sum == total_desc_in_image){
    std::cout << "Matching descriptors    " << sum << "\n";
  } else{
    std::cout << "Mismatching descriptors " << sum << "\n";
  }
}

// Create the file, as cv::FileStorage doesn't create one
void BoVW::CreateFile(const std::string &file_name)
{
  std::ofstream ostrm(file_name);
  fs::permissions(file_name,
                  fs::perms::owner_all | fs::perms::group_all,
                  fs::perm_options::add);
}

template<typename T>
void BoVW::SaveToFile(const std::string& filename, T item){
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "item" << item;
}

template<typename T>
void BoVW::LoadFromFile(const std::string& filename, T& item){
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  fs["item"] >> item;
}


void BoVW::testBoVW()
{

  //std::string test_image_filename = "/home/sgunnam/wsp/CLionProjects/myBoW/data/colombia100Objs/testdata_coil100/obj6__0.png";
  auto data_set_path = boVwParams_.dataset_dir + boVwParams_.dataset_name + boVwParams_.test_images_folder;
  ReadDirectory(data_set_path, testing_file_list);
  for (const auto &test_file : testing_file_list) {
    cv::Mat im_test = cv::imread(test_file);
    auto detector_test = cv::SIFT::create(static_cast<int>(boVwParams_.number_of_desc_per_image));
    std::vector<cv::KeyPoint> keypoints_test;
    cv::Mat descriptors_test;
    detector_test->detect(im_test, keypoints_test);
    detector_test->compute(im_test, keypoints_test, descriptors_test);
    std::vector<std::size_t> label_test;

    std::vector<std::size_t> test_hist =
      CreateTestImageHistogram(descriptors_test, centers, label_test);
    //Normalize
    hist_tfidf_ni.resize(all_hists_normalized[0].size(), 0);
    std::vector<double> test_hist_norm(test_hist.size());
    std::copy(test_hist.begin(), test_hist.end(), test_hist_norm.begin());
    auto sum_test_hist = std::accumulate(test_hist.begin(), test_hist.end(), 0ul);
    std::cout << "Normalized Histogram = \n";
    std::for_each(
      test_hist_norm.begin(),
      test_hist_norm.end(),
      [&](auto &val) {
        val = val / static_cast<double>(sum_test_hist);
        std::cout << val << ", ";
      });
    //auto test_hist_norm_check_sum = std::accumulate(test_hist_norm.begin(), test_hist_norm.end(), 0.0);
    //assert(test_hist_norm_check_sum == 1);

    PrintVecContainer("Before test_hist_norm", test_hist_norm);
    std::transform(test_hist_norm.begin(), test_hist_norm.end(), hist_tfidf_ni.cbegin(), test_hist_norm.begin(),
      [](auto &first, const auto second) { first = first*second; return first; });
    PrintVecContainer("After test_hist_norm", test_hist_norm);

    std::vector<double> ssd_vec(training_file_list.size(), 0.0);
    std::size_t file_index = 0;
    for (auto hi : all_hists_normalized) {
      ssd_vec[file_index] = std::inner_product(hi.begin(), hi.end(), test_hist_norm.begin(), 0.0,
        std::plus<>(), [](auto &l, auto &r) { return std::pow((l - r), 2); });
      std::cout << "ssd_vec[" << file_index << "] = " << ssd_vec[file_index] << std::endl;
      file_index++;
    }


    PrintVecContainer("ssd_vec : ", ssd_vec);
    auto ssd_vec_copy{ ssd_vec };
    std::sort(ssd_vec_copy.begin(), ssd_vec_copy.end());

    std::vector<std::string> matching_files;
    std::size_t file_index2 = 0;
    for (auto s : ssd_vec) {
      if (s <= ssd_vec_copy[8]) {
        matching_files.push_back(training_file_list[file_index2]);
      }
      file_index2++;
    }
    //std::copy_n(file_list.begin(), 5, matching_files.begin());
    //  for(auto i=0; i<matchCount; i++){
    //    matching_files.at(i) = file_list.at(i);
    //  }
    ShowManyImagesForBoVW("BoVW image matching", test_file, matching_files);
    std::cout << "completed processing test image " << test_file << std::endl;
//    exit(0);
  }
}

std::vector<std::size_t> BoVW::CreateTestImageHistogram(const cv::Mat& descriptors_test, const cv::Mat& centers, std::vector<std::size_t>& label_test){
  auto desc_count      = static_cast<std::size_t>(descriptors_test.rows);
  auto centroids_count = static_cast<std::size_t>(centers.rows);
  label_test.resize(desc_count, SIZE_MAX);
  for(auto i=0ul; i<desc_count; i++){
    auto euclidean_ssd = DBL_MAX;
    for(auto j=0ul; j<centroids_count; j++){
      auto v1 = descriptors_test.row(static_cast<int>(i));
      auto v2 = centers.row(static_cast<int>(j));
      auto euclidean_dist = ((cv::abs(v1-v2)));
      cv::MatExpr euclidian_dist_sqr = euclidean_dist.mul(euclidean_dist);
      if (sum(euclidian_dist_sqr)[0] < euclidean_ssd) {
        euclidean_ssd = sum(euclidian_dist_sqr)[0];
        label_test[i] = j;
      }

    }
  }

  // generate histogram
  std::vector<std::size_t> test_hist(static_cast<std::size_t>(centers.size[0]), 0);
  for(auto lab:label_test)
    ++test_hist.at(lab);
  return test_hist;
}


using namespace cv;
using namespace std;

/*Function///////////////////////////////////////////////////////////////

Name:       ShowManyImages

Purpose:

This is a function illustrating how to display more than one
image in a single window using Intel OpenCV

Parameters:

string title: Title of the window to be displayed
int    nArgs: Number of images to be displayed
Mat    img1: First Mat, which contains the first image
...
Mat    imgN: First Mat, which contains the Nth image

Language:   C++

The method used is to set the ROIs of a Single Big image and then resizing
and copying the input images on to the Single Big Image.

This function does not stretch the image...
It resizes the image without modifying the width/height ratio..

This function can be called like this:

ShowManyImages("Images", 5, img2, img2, img3, img4, img5);

This function can display upto 12 images in a single window.
It does not check whether the arguments are of type Mat or not.
The maximum window size is 700 by 660 pixels.
Does not display anything if the number of arguments is less than
one or greater than 12.

Idea was from [[BettySanchi]] of OpenCV Yahoo! Groups.

If you have trouble compiling and/or executing
this code, I would like to hear about it.

You could try posting on the OpenCV Yahoo! Groups
[url]http://groups.yahoo.com/group/OpenCV/messages/ [/url]


Parameswaran,
Chennai, India.

cegparamesh[at]gmail[dot]com

...
///////////////////////////////////////////////////////////////////////*/
// TO Do, Modify to accommodate multiple types of channel
void BoVW::ShowManyImagesForBoVW(const std::string& title, const std::string& test_image, const std::vector<std::string>& matching_images) {

  auto nImages = 1 + matching_images.size();
  std::vector<cv::Mat> images_list(nImages);
  images_list.at(0) = cv::imread(test_image);
  for(auto i=1ul; i < (nImages); i++){
    images_list.at(i) = cv::imread(matching_images.at(i-1));
  }

  int size;
  int i;
  int m, n;
  int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
  int w, h;

// scale - How much we have to resize the image
  float scale;
  int max;

// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
  if(nImages <= 0) {
    printf("Number of arguments too small....\n");
    return;
  }
  else if(nImages > 14) {
    printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
    return;
  }
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
  else if (nImages == 1) {
    w = h = 1;
    size = 300;
  }
  else if (nImages == 2) {
    w = 2; h = 1;
    size = 300;
  }
  else if (nImages == 3 || nImages == 4) {
    w = 2; h = 2;
    size = 300;
  }
  else if (nImages == 5 || nImages == 6) {
    w = 3; h = 2;
    size = 200;
  }
  else if (nImages == 7 || nImages == 8) {
    w = 4; h = 2;
    size = 200;
  }
  else {
    w = 4; h = 3;
    size = 150;
  }

// Create a new 3 channel image
  Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);


// Loop for nImages number of arguments
  for (i = 0, m = 20, n = 20; i <  static_cast<int>(nImages); i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    Mat img = images_list.at( static_cast<unsigned long>(i));

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if(img.empty()) {
      printf("Invalid arguments");
      return;
    }

    // Find the width and height of the image
    x = img.cols;
    y = img.rows;

    // Find whether height or width is greater in order to resize the image
    max = (x > y)? x: y;

    // Find the scaling factor to resize the image
    scale =  ( static_cast<float>( max) / static_cast<float>(size));

    // Used to Align the images
    if( i % w == 0 && m!= 20) {
      m = 20;
      n+= 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    Rect ROI(m, n,  static_cast<int>( static_cast<float>(x)/scale ),
             static_cast<int>( static_cast<float>(y)/scale ));
    Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
  }

// Create a new window, and show the Single Big Image
  namedWindow( title, 1 );
  imshow( title, DispImage);
  waitKey();
}