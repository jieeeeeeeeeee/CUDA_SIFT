#ifndef CUSIFT_FUNCTION_H_H
#define CUSIFT_FUNCTION_H_H

#include "cuImage.h"
#include "cuGlobal.h"
//#include "cusitf_function_D.h"

using namespace cv;



cv::Mat cv::getGaussianKernel( int n, double sigma, int ktype );

void disMatf(cuImage &cuImg);



extern "C"
void useCUDA();



extern "C"
void cuGaussianBlur(cuImage& cuImg,float sigma);


#endif
