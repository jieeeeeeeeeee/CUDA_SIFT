cmake_minimum_required(VERSION 2.8.12)

project(cuda_sift)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#find qt5
find_package(Qt5Core)

#find OpenCV
set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} Qt5::Core)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
