/*
 *  This is a test for GaussianBlur
 *
 *
 */


#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
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
    src = imread("../data/road.png",0);
    //src = imread("../data/DSC04034.JPG",0);
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
    Mat tmp;
    src.convertTo(tmp, CV_32FC1);

#if TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    GaussianBlur(src,dst,Size(0,0),2);
#if TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif

////////////////////////////////////////////////////
//// old cuImage
///////////////////////////////////////////////////
//#if TIME
//    t = (double)getTickCount();
//#endif
//    cuImage cuimg;
//    cuimg.AllocateMat1D(src);
//    cuGaussianBlur(cuimg,1.22);

////    Mat dis(cuimg.height,cuimg.width,CV_32F);
////    memcpy(dis.data,cuimg.h_data,cuimg.width*cuimg.height*sizeof(float));
////    dis.convertTo(dst,DataType<uchar>::type, 1, 0);
//#if TIME
//    t = (double)getTickCount() - t;
//    printf("time cost: %g ms\n", t*1000./tf);
//#endif

////////////////////////////////////////////////////
//// new cudaImage
///////////////////////////////////////////////////
#if TIME
    t = (double)getTickCount();
#endif

    CudaImage cuimg;
    cuimg.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)tmp.data);
    cuimg.Download();
    cuGaussianBlur(cuimg,2);

//        Mat dis(cuimg.height,cuimg.width,CV_32F);
//        memcpy(dis.data,cuimg.h_data,cuimg.width*cuimg.height*sizeof(float));
//        dis.convertTo(dst,DataType<uchar>::type, 1, 0);
#if TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif


    cvNamedWindow("GaussBlar",CV_WINDOW_NORMAL);
    imshow("GaussBlar",dst);
    waitKey(0);
    return 0;
}
