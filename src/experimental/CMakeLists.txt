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
