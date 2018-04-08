/*
 *  This is a program testing for GaussianBlur
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

#define TIME
#define IMAGE_SHOW
#define CHECK_RESIDUAL

int main()
{
    //useCUDA();
    Mat src;
    //src = imread("../data/road.png",0);
    src = imread("../data/DSC04034.JPG",0);
    //src = imread("../data/100_7101.JPG",0);
    //src = imread("../data/lena.png",0);
    //tmp.convertTo(src, CV_32FC1);

    int sigma = 2;

    Mat dst;
    int width = src.cols;
    int height = src.rows;
    if(!src.data)
    {
        printf("no photo");
    }
    Mat tmp;
    src.convertTo(tmp, CV_32FC1);

#ifdef TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    GaussianBlur(src,dst,Size(0,0),sigma);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif


////////////////////////////////////////////////////
//// new cudaImage
///////////////////////////////////////////////////
#ifdef TIME
    t = (double)getTickCount();
#endif

    CudaImage cuimg;
    cuimg.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)tmp.data);
    cuimg.Download();
    cuGaussianBlur(cuimg,sigma);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g ms\n", t*1000./tf);
#endif





#ifdef IMAGE_SHOW
    Mat dis(cuimg.height,cuimg.width,CV_32F);
    safeCall(cudaMemcpy2D(dis.data,cuimg.width*sizeof(float),cuimg.d_data,cuimg.pitch*sizeof(float),cuimg.width*sizeof(float),(size_t) cuimg.height,cudaMemcpyDeviceToHost));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 0);
    cvNamedWindow("ss",CV_WINDOW_NORMAL);
    imshow("ss",gray);
    waitKey(0);

    cvNamedWindow("GaussBlar",CV_WINDOW_NORMAL);
    imshow("GaussBlar",dst);
    waitKey(0);
#endif




#ifdef CHECK_RESIDUAL
    Mat residualMat(dst.size(),dst.type());

    for(int i = 0;i<dis.rows;i++)
    {
        uchar *p = gray.ptr<uchar>(i);
        uchar *q = dst.ptr<uchar>(i);
        uchar *d = residualMat.ptr<uchar>(i);
        for(int j = 0;j<dis.cols;j++){
            d[j] = abs(p[j] - q[j]) * 30;
        }
    }
    cvNamedWindow("residual",CV_WINDOW_NORMAL);
    imshow("residual",residualMat);
    waitKey(0);
#endif

    return 0;
}
