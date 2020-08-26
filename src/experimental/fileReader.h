//
// Created by srigun on 8/25/20.
//
#include<iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#ifndef MYBOW_SRC_EXPERIMENTAL_FILEREADER_H_
#define MYBOW_SRC_EXPERIMENTAL_FILEREADER_H_

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

#endif//MYBOW_SRC_EXPERIMENTAL_FILEREADER_H_
