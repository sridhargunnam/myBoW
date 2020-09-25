# Generic test that uses conan libs

add_executable(experimental main.cpp customFilter.cpp customFilter.h fileReader.h ImageHelper.h)
set_property(TARGET experimental PROPERTY CXX_STANDARD 17)


target_link_libraries(experimental
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})

add_executable(fileStorage1 fileStorage.cpp)
target_link_libraries(fileStorage1
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS}
        gtest
        gmock)
	#benchmark::benchmark)


add_executable(TestStats TestStats.cpp Stats.cpp Stats.h)
target_link_libraries(TestStats
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})

add_library(bowStats Stats.cpp Stats.h)
target_link_libraries(bowStats
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})



add_executable(TestRealSense TestRealSense.cpp TestRealSense.cpp)
target_link_libraries(TestRealSense
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS}
        ${realsense_INCLUDE_DIR})

add_library(libBoVW BoVW.cpp BoVW.h showMultipleImages.h)
target_link_libraries(libBoVW
        PRIVATE bowStats
        project_options
        project_warnings
        ${OpenCV_LIBS}
        gtest
        gmock)

# Generic test that uses conan libs
add_executable(testBoVW
        testBoVW.cpp)
target_include_directories(testBoVW PRIVATE experimental)
target_link_libraries(testBoVW
        PRIVATE
        libBoVW)