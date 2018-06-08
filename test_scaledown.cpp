#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>

#define TEST_OTHER_SCALEDOWN


//#define IMAGE_SHOW
using namespace cv;
using namespace std;
Mat disMatf(CudaImage &cuImg,char *str)
{
    Mat dis(cuImg.height,cuImg.width,CV_32F);

    for(int i = 0;i<dis.rows;i++)
    {
        float *p = dis.ptr<float>(i);
        for(int j = 0;j<dis.cols;j++){
            p[j] = cuImg.h_data[i*dis.cols+j];
            //std::cout<<p[j]<<" ";
        }
        //std::cout<<std::endl;
    }
    //memcpy(dis.data,cuImg.h_data,cuImg.width*cuImg.height*sizeof(float));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 0);

    cvNamedWindow(str,CV_WINDOW_NORMAL);
    imshow(str,gray);
    waitKey();
    return gray;
}
void computeResidual(Mat&gray,Mat &dst,char *str = "residual")
{
    Mat residualMat(dst.size(),dst.type());

    for(int i = 0;i<dst.rows;i++)
    {
        uchar *p = gray.ptr<uchar>(i);
        uchar *q = dst.ptr<uchar>(i);
        uchar *d = residualMat.ptr<uchar>(i);
        for(int j = 0;j<dst.cols;j++){
            d[j] = abs(p[j] - q[j]) * 30;
        }
    }
    cvNamedWindow(str,CV_WINDOW_NORMAL);
    imshow(str,residualMat);
    waitKey(0);
}


int main()
{
    //char *a ="../data/road.png";
    char *a ="../data/lena.png";
    //char *a ="../data/100_7101.JPG";
    cv::Mat src;
    src = imread("../data/100_7101.JPG",0);
    //src = imread("../data/lena.png",0);
    //src = imread(a,0);
    //src = imread("../data/DSC04034.JPG",0);
    int width = src.cols;
    int height = src.rows;
    if(!src.data)
    {
        printf("no photo");
    }
    int sigma1 = 1.6;

    Mat tmp,downtmp;
    src.convertTo(tmp, CV_32FC1);
    resize(tmp, downtmp, Size(src.cols/2, src.rows/2),
           0, 0, INTER_NEAREST);
    //warming up time
#ifdef TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    CudaImage cuimg;
    cuimg.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)tmp.data);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("time cost in warming up: %g ms\n", t*1000./tf);
#endif

    //host to device
    cuimg.Download();

    /*for normal imgae*/
//    cuimg.Readback();
//    //show device image
//    Mat dst;
//    dst = disMatf(cuimg,"device");

//    //show host image
//    Mat gray;
//    tmp.convertTo(gray,DataType<uchar>::type, 1, 0);
//    cvNamedWindow("host",CV_WINDOW_NORMAL);
//    imshow("host",gray);
//    waitKey();

//    computeResidual(gray,dst);
    /*for normal imgae*/



#ifdef TEST_OTHER_SCALEDOWN
    CudaImage subImg;
    int p = iAlignUp(cuimg.width/2, 128);
    subImg.Allocate(cuimg.width/2, cuimg.height/2, p, true);
    ScaleDown(subImg, cuimg, 0.5f);
    subImg.Readback();
    Mat dst;
    dst = disMatf(subImg,"device");

    Mat gray;
    downtmp.convertTo(gray,DataType<uchar>::type, 1, 0);
    cvNamedWindow("host",CV_WINDOW_NORMAL);
    imshow("host",gray);
    waitKey();

    computeResidual(gray,dst);
#endif

    return 0;
}
