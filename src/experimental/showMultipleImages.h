//
// Created by srigun on 6/26/20.
//

#ifndef EXAMPLES_SHOWMULTIPLEIMAGES_H
#define EXAMPLES_SHOWMULTIPLEIMAGES_H
#pragma once
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdarg.h>

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
#include <algorithm>
// TO Do, Modify to accommodate multiple types of channel
void ShowManyImagesForBoVW(const std::string& title, const std::string& test_image, const std::vector<std::string>& matching_images) {

  auto nImages = 1 + matching_images.size();
  std::vector<cv::Mat> images_list(nImages);
  images_list.at(0) = cv::imread(test_image);
  for(auto i=1ul; i < (nImages); i++){
    images_list.at(i) = cv::imread(matching_images.at(i-1));
  }
//  std::transform(matching_images.begin(), matching_images.end(), images_list.end(),
//                  [](const auto& file, auto& im){im = cv::imread(file); return im;});

  //  cv::namedWindow("ShowManyImages", cv::WINDOW_NORMAL);
//  for(auto i=0; i<(nImages); i++){
//    int rows = images_list.at(i).rows;
//    int cols = images_list.at(i).cols;
//    int output_width = 700;
//    float scaling_ratio = (static_cast<float>(rows)/output_width);
//    auto dest_size = cv::Size(round(cols/scaling_ratio), output_width);
//    cv::resize(images_list[i], images_list[i], dest_size);
//    cv::imshow("Image", images_list[i]);
//    cv::waitKey(0);
//  }

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

#endif //EXAMPLES_SHOWMULTIPLEIMAGES_H
