//
// Created by sgunnam on 9/23/20.
//
#include "BoVW.h"
int main(){
  BoVWParams boVwParams;
  /*{
    20,
  500,
      "/home/sgunnam/wsp/CLionProjects/myBoW/data",
    "/myRoom",
    "/training",
    "/test_images",
    "/debug_images",
    "/save_dir",
  } ; */
  BoVW b{boVwParams};
  return 0;
}

// TODO
// 1) Read Images with exception
// 2) Handle Zero descriptors for an image
      //descriptor count = 0 for image /home/sgunnam/wsp/CLionProjects/myBoW/data/smallImagedataset/training/obj35__25.png

/*
ssd_vec[0] = 949.910
ssd_vec[1] = 1328.257
ssd_vec[2] = 1110.115
ssd_vec[3] = 5316.300
ssd_vec[4] = 1849.745
ssd_vec[5] = 4479.459
ssd_vec[6] = 615.588
ssd_vec[7] = 1203.700
ssd_vec[8] = 3348.066
ssd_vec[9] = 1498.198
ssd_vec[10] = 2738.275
ssd_vec[11] = 2579.891
ssd_vec[12] = 1973.042
ssd_vec[13] = 1802.187
ssd_vec[14] = 2685.925
 */