#include <iostream>
#include "sift/sift.h"

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;  //包含cv命名空间
using namespace std;

#define TIME 0



int main()
{

#if TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif


    //Create SIFT class pointer
    Ptr<Feature2D> f2d = xfeatures2d::q::SIFT::create();
    //读入图片
    Mat img_1 = imread("../data/img2.ppm");
    Mat img_2 = imread("../data/img3.ppm");
    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);
    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);
    //Matching descriptor vector using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    //绘制匹配出的关键点
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    cvNamedWindow("match",CV_WINDOW_NORMAL);
    imshow("match", img_matches);
    //等待任意按键按下


#if TIME
    t = (double)getTickCount() - t;
    printf("time cost: %g\n", t*1000./tf);
#endif

    waitKey(0);
}

//#include "opencv2/core.hpp"
//#include "opencv2/cudaarithm.hpp"
//#include "opencv2/cudafilters.hpp"

//int main()
//{
//    Mat src;
//    src = imread("../data/100_7106.JPG",0);

//    cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(src.type(), CV_32F, Size(0,0), 2.5, 2.5, cv::BORDER_DEFAULT);

//    cv::cuda::GpuMat dst ;
//    dst.create(src.size(), CV_32F);
//    gauss->apply(src, dst);



//    return 0;
//}
