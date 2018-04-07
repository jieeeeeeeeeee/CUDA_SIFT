project(cusift)


find_package(CUDA)

set(CUDA_NVCC_FLAGS -arch=compute_60)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)


source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

#SET(CMAKE_BUILD_TYPE "Debug")


message(STATUS "This is src file:${OpenCV_LIBS}")




cuda_add_library(cusift SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#target_link_libraries(cusift ${CUDA_LIBRARIES})



