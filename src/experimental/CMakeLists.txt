# Generic test that uses conan libs

add_executable(experimental main.cpp)
set_property(TARGET experimental PROPERTY CXX_STANDARD 17)


target_link_libraries(experimental
        PRIVATE project_options
        project_warnings
        CONAN_PKG::docopt.cpp
        CONAN_PKG::fmt
        CONAN_PKG::spdlog
        ${OpenCV_LIBS}
        gtest
        gmock
        benchmark::benchmark
        ${realsense_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS})