#ifndef CUSIFT_FUNCTION_H
#define CUSIFT_FUNCTION_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;

extern "C"
void useCUDA();


extern "C"
void cuGaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY, int borderType);

#endif
