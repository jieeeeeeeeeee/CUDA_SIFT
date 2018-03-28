cmake_minimum_required(VERSION 2.8.12)

project(cuda_sift)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(SRC main.cpp)
set(EXTRA_SRC)

#find qt5
find_package(Qt5Core)

#find OpenCV
#which is release version
#set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build)

#OpenCV Labrary which is debug version
set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build2)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_subdirectory(sift)
add_subdirectory(cuda)

SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS "-pg")

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} Qt5::Core)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} sift)
target_link_libraries(${PROJECT_NAME} cusift)

option (BUILD_TEST "build test" OFF)
if (BUILD_TEST)
  set (EXTRA_SRC ${EXTRA_SRC} test_GaussianBlur.cpp)
  #add_executable(test_GaussianBlur ${EXTRA_SRC})
endif (BUILD_TEST)

