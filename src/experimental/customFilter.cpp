//
// Created by srigun on 8/25/20.
//

#include "customFilter.h"

[[maybe_unused]] customFilter::customFilter(const std::string& filename){
  // Declare variables
  cv::Mat src, dst;
  const char* window_name = "filter2D Demo";;
  // Loads an image
  src = imread( cv::samples::findFile( filename ), cv::IMREAD_COLOR ); // Load an image
  if( src.empty() )
  {
    printf(" Error opening image\n");
    //return EXIT_FAILURE;
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
  //return EXIT_SUCCESS;
}
