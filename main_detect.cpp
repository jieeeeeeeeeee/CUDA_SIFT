#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>

#define USE_MY_SIFT


#ifdef USE_SIFT OR USE_SURF
#include "opencv2/features2d.hpp"
#endif
#ifdef USE_MY_SIFT
#include"sift/sift.h"
#endif



//#define IMAGE_SHOW
using namespace cv;
using namespace std;

void siftdect(cv::Mat& src,std::vector<cv::KeyPoint> keypoints){

#ifdef NODOUBLEIMAGE
    int firstOctave = 0, actualNOctaves = 0, actualNLayers = 0;
#else
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
#endif


#ifdef FIND_DOGERRORTEST
#else
    CudaImage base;
#endif
    createInitialImage(src,base,(float)1.6,firstOctave<0);

    int nOctaveLayers = 3;
#ifdef TEST_FIRST_OCTAVE
    int nOctaves = cvRound(std::log( (double)std::min( base.width, base.height ) ) / std::log(2.) - 8) - firstOctave;
#else
    int nOctaves = cvRound(std::log( (double)std::min( base.width, base.height ) ) / std::log(2.) - 2) - firstOctave;
#endif
    std::vector<CudaImage> gpyr,dogpyr;
    buildGaussianPyramid(base, gpyr, nOctaves);
    buildDoGPyramid(gpyr, dogpyr);
    float* h_keypoints;
    findScaleSpaceExtrema(gpyr, dogpyr, h_keypoints);

}

#define TIME


int main()
{
    std::cout<<"Hello World !"<<std::endl;
    //char *a ="../data/img2.ppm";
    //char *a ="../data/road.png";
    //char *a ="../data/lena.png";
    char *a ="../data/100_7101.JPG";
    //char *a ="../data/DSC04034.JPG";
    //char *a ="../data/1080.jpg";
    cv::Mat src;
    src = imread(a);

//    resize(src,src,Size(src.cols*2,src.rows*2),0,0);
//    Mat gray;
//    src.convertTo(gray,DataType<uchar>::type);
//    namedWindow("ss",CV_WINDOW_NORMAL);
//    imshow("ss",gray);
//    waitKey(0);

    int width = src.cols;
    int height = src.rows;
    if(!src.data)
    {
        printf("no photo");
    }

    Mat tmp;
    src.convertTo(tmp, CV_32FC1);

    std::vector<cv::KeyPoint> keypoints;
#ifdef TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    siftdect(src,keypoints);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("first cost : %g ms\n", t*1000./tf);//246ms
#endif


#ifdef TIME
    t = (double)getTickCount();
#endif
    siftdect(src,keypoints);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("second cost : %g ms\n", t*1000./tf);//158
#endif

    /////////////////////////////////////
    /// SIFT
    /////////////////////////////////////
    //Create SIFT class pointer
//#ifdef USE_MY_SIFT
//    Ptr<Feature2D> f2d = xfeatures2d::q::SIFT::create();
//#else
//    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//#endif
//    //读入图片
//    Mat img_1 = src;

//    //Detect the keypoints
//    vector<KeyPoint> keypoints_1, keypoints_2;

//#ifdef TIME
//    t = (double)getTickCount();
//#endif
//    f2d->detect(img_1, keypoints_1);
//#ifdef TIME
//    t = (double)getTickCount() - t;
//    printf("opencv sift cost : %g ms\n", t*1000./tf);//158
//#endif


//    //Calculate descriptors (feature vectors)
//    Mat descriptors_1, descriptors_2;
//    f2d->compute(img_1, keypoints_1, descriptors_1);

//    std::cout<<"sift keypoints num :"<<keypoints_1.size()<<std::endl;
//    Mat kepoint;
//    drawKeypoints(img_1, keypoints_1,kepoint,cv::Scalar::all(-1),4);
//    cvNamedWindow("extract",CV_WINDOW_NORMAL);
//    imshow("extract", kepoint);
//    //等待任意按键按下
//    waitKey(0);

//    for(int i = 0;i < keypoints_1.size();i++)
//        std::cout<<keypoints_1[i].pt.x<<" ";
//    std::cout<<std::endl;

#ifdef COMPARE_VALUE

    sort(keypoints_1.begin(),keypoints_1.end(),sortx);
    int unique_nums;
    unique_nums = std::unique(keypoints_1.begin(),keypoints_1.end(),uniquex) - keypoints_1.begin();
    for(int i = 0;i < unique_nums;i++)
        std::cout<<keypoints_1[i].response<<" ";
    std::cout<<unique_nums<<std::endl;

#endif



    return 0;
}
