# Generic test that uses conan libs

add_executable(experimental main.cpp customFilter.cpp customFilter.h fileReader.h ImageHelper.h)
set_property(TARGET experimental PROPERTY CXX_STANDARD 17)


target_link_libraries(experimental
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})