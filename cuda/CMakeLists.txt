project(cusift)


find_package(CUDA)

set(CUDA_NVCC_FLAGS -arch=compute_60;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)


source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})


message(STATUS "This is src file:${CURRENT_HEADERS}")


cuda_add_library(cusift SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})




