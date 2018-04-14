cmake_minimum_required(VERSION 2.8.12)

project(cuda_sift)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(SRC main.cpp)
set(EXTRA_SRC
test_GaussianBlur.cpp
)

find_package(CUDA)
#find qt5
find_package(Qt5Core)

#find OpenCV
#which is release version
#set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build)

#OpenCV Labrary which is debug version build2
set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build1)
#set(OpenCV_DIR /home/jie/third_party/opencv-3.4.0/build3)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(sift)
add_subdirectory(cuda)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS "-pg")
set(LABRARY ${OpenCV_LIBS} "Qt5::Core"  "sift" "cusift")

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} ${LABRARY})
#message(STATUS "This is src file:${LABRARY}")
#target_link_libraries(${PROJECT_NAME} Qt5::Core)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(${PROJECT_NAME} sift)
#target_link_libraries(${PROJECT_NAME} cusift)

option (BUILD_TEST "build test" ON)
if (BUILD_TEST)
  set (EXTRA_SRC ${EXTRA_SRC} )
  #cuda_add_executable(test_GaussianBlur ${EXTRA_SRC})

  add_executable(test_GaussianBlur ${EXTRA_SRC})
  target_link_libraries(test_GaussianBlur ${LABRARY})

  add_executable(test_MySIFT test_MySIFT.cpp)
  target_link_libraries(test_MySIFT ${LABRARY})

  add_executable(cuda_main cuda_main.cpp)
  target_link_libraries(cuda_main ${LABRARY})

endif (BUILD_TEST)

