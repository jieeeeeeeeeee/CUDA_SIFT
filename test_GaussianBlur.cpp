/*
 *  This is a test for GaussianBlur
 *
 *
 */


#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;  //包含cv命名空间
using namespace std;

#define TIME 1


int main()
{
    //useCUDA();
    Mat src;
    src = imread("../data/100_7106.JPG",0);
    //tmp.convertTo(src, CV_32FC1);

//    Mat ss;
//    src.convertTo(ss,CV_8UC1);
//    float * h_data = (float*)src.data;

//    for(int i= 0 ;i<src.rows;i++)
//        for(int j = 0;j<src.cols;j++){
//            std::cout<<h_data[i*src.rows+j]<<std::endl;
//        }


    Mat dst;
    int width = src.cols;
    int height = src.rows;
    if(!src.data)
    {
        printf("no photo");
    }

#if TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    GaussianBlur(src,dst,Size(0,0),1.6);
#if TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif



#if TIME
    t = (double)getTickCount();
#endif
    cuImage cuimg;
    cuimg.AllocateMat1D(src);
    cuGaussianBlur(cuimg,1.6);
#if TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif



    cvNamedWindow("GaussBlar",CV_WINDOW_NORMAL);
    imshow("GaussBlar",dst);
    waitKey(0);
    return 0;
}
