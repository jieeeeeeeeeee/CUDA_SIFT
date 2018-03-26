
project(sift)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 
SET(CMAKE_BUILD_TYPE "Debug")


add_library(sift SHARED ${CURRENT_HEADERS} sift.cpp)
