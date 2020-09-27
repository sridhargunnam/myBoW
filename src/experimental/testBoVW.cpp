//
// Created by sgunnam on 9/23/20.
//
#include "BoVW.h"
int main(){
  BoVWParams boVwParams;
  BoVW b{boVwParams};
  return 0;
}

// TODO
//  Performance : Profile, Parallelize the code
//  Validation using metrics like accuracy(partially done), the usefulness of centroid
//  Refactor for performance, testing and debug capabilities(be able to annotate descriptors, keypoints at later stages of the pipeline
//  Test - Reading Images with no descriptors, file checking
//  Capabilities - change descriptors
//  Profile for various parameter changes
//  Reuse saved data if available already - modify save/load data operations
//descriptor count = 0 for image /home/sgunnam/wsp/CLionProjects/myBoW/data/smallImagedataset/training/obj35__25.png
