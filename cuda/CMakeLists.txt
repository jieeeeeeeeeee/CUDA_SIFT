project(cusift)

find_package(CUDA)

#set(CUDA_NVCC_FLAGS -arch=sm_61;-Xptxas=-v)
#set(CUDA_NVCC_FLAGS -arch=compute_60;-G;-g;-maxrregcount=32)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

SET(SRC
    cusitf_function_H.cu;
    cudaImage.cu
    cuSIFT.cpp
    cuSIFT.cu
    )
set(cuda_sources
    cuGlobal.h
    cudaImage.cu
    cusitf_function_D.h
    cusitf_function_H.h
    cusitf_function_H.cu
    cuSIFT_H.cu
    )

SET(CMAKE_BUILD_TYPE "Debug")

#message(STATUS "This is src file:${CURRENT_SOURCES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(cusift SHARED  ${cuda_sources}  )
#cuda_add_library(cusift SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#target_link_libraries(cusift ${CUDA_LIBRARIES})



