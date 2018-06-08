#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>

#define USE_MY_SIFTs

#ifndef USE_MY_SIFT
#include "opencv2/features2d.hpp"
#endif
#ifdef USE_MY_SIFT
#include"sift/sift.h"
#endif

//#define IMAGE_SHOW
using namespace cv;
using namespace std;
int findSamePointsIndex(cv::KeyPoint& keypoint,std::vector<cv::KeyPoint>&keypoints);
static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}


bool uniquex(const KeyPoint& a,const KeyPoint& b)
{
    bool x;
    if(a.pt.x == b.pt.x && a.pt.y == b.pt.y)
        x = true;
    else
        x = false;
    return x;
}
bool sortx(const KeyPoint& a,const KeyPoint& b)
{
    bool x;
    if(a.pt.x < b.pt.x)
        x = true;
    else
        x = false;
    return x;
}

int evaluateDetector(std::vector<KeyPoint> &one,std::vector<KeyPoint> &another,int stride=3){

    int sum=0;
    int offset=0;
    for(int i = 0;i<one.size();i++)
    {
        KeyPoint x,y;
        x = one[i];
        int index = i<another.size()?i:another.size()-1;
        y = another[index];
        if(x.pt.x==y.pt.x&&x.pt.y==y.pt.y)
            sum+=1;
        else{
            for(int j = -stride;j<stride;j++)
            {
                int idx = i+j;
                if(idx>0 && idx<another.size()-1)
                    y = another[idx];
                if(x.pt.x==y.pt.x&&x.pt.y==y.pt.y)
                    sum+=1;
            }
        }

    }
    return sum;
}

int evaluateDetectorBuforce(std::vector<KeyPoint> &one,std::vector<KeyPoint> &another){

    int sum=0;

    for(int i = 0;i<one.size();i++)
    {
        KeyPoint x,y;
        x = one[i];
        for(int j = 0;j<another.size();j++)
        {
            y = another[j];
            if(x.pt.x==y.pt.x&&x.pt.y==y.pt.y)
            {
                sum++;
                break;
            }
        }
        if(i % 1000 == 0)
        {
            std::cout<<"run in:"<<i<<std::endl;
        }
    }
    return sum;
}

int evaluateDetectorBuforce(std::vector<KeyPoint> &one,int size,std::vector<KeyPoint> &another,int size1,float e){

    int sum=0;

    for(int i = 0;i<size;i++)
    {
        KeyPoint x,y;
        x = one[i];
        for(int j = 0;j<size1;j++)
        {
            y = another[j];
            if(abs(x.pt.x-y.pt.x)<e&&abs(x.pt.y-y.pt.y)<e)
            {
                sum++;
                break;
            }
        }
        if(i % 1000 == 0)
        {
            std::cout<<"run in:"<<i<<std::endl;
        }
    }
    return sum;
}



void siftdect(cv::Mat& src,std::vector<cv::KeyPoint>& keypoints,cv::Mat& descriptors){

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


    findScaleSpaceExtrema(gpyr, dogpyr,keypoints,descriptors);

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
    cv::Mat descriptors;
#ifdef TIME
    double t, tf = getTickFrequency();
    t = (double)getTickCount();
#endif
    siftdect(src,keypoints,descriptors);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("first cost : %g ms\n", t*1000./tf);//246ms
#endif



    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
#ifdef TIME
    t = (double)getTickCount();
#endif
    siftdect(src,keypoints1,descriptors1);
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("second cost : %g ms\n", t*1000./tf);//158
#endif

    /////////////////////////////////////
    /// SIFT
    /////////////////////////////////////
    //Create SIFT class pointer
#ifdef USE_MY_SIFT
    Ptr<Feature2D> f2d = xfeatures2d::q::SIFT::create();
#else
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
#endif
    //读入图片
    Mat img_1 = src;

    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;

#ifdef TIME
    t = (double)getTickCount();
#endif
    f2d->detect(img_1, keypoints_1);

    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);

#ifdef TIME
    t = (double)getTickCount() - t;
    printf("opencv sift cost : %g ms\n", t*1000./tf);//158
#endif

//    /************* show *******************/

//    std::cout<<"sift keypoints num :"<<keypoints_1.size()<<std::endl;
//    Mat kepointImg_sift;
//    drawKeypoints(img_1, keypoints_1,kepointImg_sift,cv::Scalar::all(0),4);
//    cvNamedWindow("kepointImg_sift",CV_WINDOW_NORMAL);
//    imshow("kepointImg_sift", kepointImg_sift);
//    //等待任意按键按下
//    //waitKey(0);

//    std::vector<cv::KeyPoint> keypoints2;
//    Mat kepointImg_cu;
//    drawKeypoints(img_1, keypoints1,kepointImg_cu,cv::Scalar::all(255),4);
//    cvNamedWindow("kepointImg_cu",CV_WINDOW_NORMAL);
//    imshow("kepointImg_cu", kepointImg_cu);
//    //等待任意按键按下
//    //waitKey(0);


//    //cudasift cover origonal sift
//    Mat cuCoversift;
//    drawKeypoints(kepointImg_sift, keypoints1,cuCoversift,cv::Scalar::all(255),4);
//    cvNamedWindow("cuCoversift",CV_WINDOW_NORMAL);
//    imshow("cuCoversift", cuCoversift);
//    //等待任意按键按下
//    //waitKey(0);

//    //origonal sift cover cudasift
//    Mat siftCovercu;
//    drawKeypoints(kepointImg_cu, keypoints_1,siftCovercu,cv::Scalar::all(0),4);
//    cvNamedWindow("siftCovercu",CV_WINDOW_NORMAL);
//    imshow("siftCovercu", siftCovercu);
//    //等待任意按键按下
//    waitKey(0);


//    /************ compare ***************/

//    sort(keypoints_1.begin(),keypoints_1.end(),sortx);
//    int unique_nums,unique_nums1;
//    unique_nums = std::unique(keypoints_1.begin(),keypoints_1.end(),uniquex) - keypoints_1.begin();
////    for(int i = 0;i < unique_nums;i++)
////        std::cout<<keypoints_1[i].response<<" ";
//    std::cout<<unique_nums<<std::endl;


//    sort(keypoints1.begin(),keypoints1.end(),sortx);
//    //int unique_nums;
//    unique_nums1 = std::unique(keypoints1.begin(),keypoints1.end(),uniquex) - keypoints1.begin();
////    for(int i = 0;i < unique_nums;i++)
////        std::cout<<keypoints1[i].response<<" ";
//    std::cout<<unique_nums1<<std::endl;

//    int sameCount = evaluateDetectorBuforce(keypoints_1,unique_nums,keypoints1,unique_nums1,0.2);

//    std::cout<<"sameCount:"<<sameCount<<" rate:"<<(float)sameCount/unique_nums<<std::endl;
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


#define TEST_DESCRIPTORs
#ifdef TEST_DESCRIPTOR
    int k = 0;
    std::map<int,int> map;
    for(int i = 0;i<keypoints_1.size();i++)
    {
        int idx = findSamePointsIndex(keypoints_1[i],keypoints);

        if(idx){
            //printf("%d -- %d \n",i,idx);
            map.insert(std::pair<int,int>(i,idx));
            k++;
        }
    }
    //printf("k: %d -- %d \n",k,(int)keypoints_1.size());
    if(keypoints_1.size()==k)
        printf("all match !");
    else
        printf("not all match !");

    cv::Mat difImg;
    difImg.create(k,128,CV_8UC1);
    memset(difImg.data,0,difImg.cols*difImg.rows*sizeof(uchar));


    std::map<int,int>::iterator iter;
    int i = 0;
    for(iter = map.begin();iter!=map.end();iter++)
    {
        float* psift = descriptors_1.ptr<float>(iter->first);
        float* pcuda = descriptors.ptr<float>(iter->second);
        uchar* dif = difImg.ptr<uchar>(i);
        for(int j = 0;j<difImg.cols;j++){
            dif[j] = std::abs(psift[j] - pcuda[j])*50;
            if(dif[j]>100){
                KeyPoint kpt = keypoints_1[iter->first];
                int octave, layer;
                float scale;
                unpackOctave(kpt, octave, layer, scale);
                Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
                printf("x:%f,y:%f,angle: %f\n",ptf.x,ptf.y,kpt.angle);
            }
        }
        i++;
    }

    cvNamedWindow("dif",CV_WINDOW_NORMAL);
    imshow("dif", difImg);
    waitKey(0);

#endif

    return 0;
}
int findSamePointsIndex(cv::KeyPoint& keypoint,std::vector<cv::KeyPoint>&keypoints){

    for(int i = 0;i<keypoints.size();i++)
    {
        if(keypoint.pt == keypoints[i].pt && std::abs(keypoint.angle-keypoints[i].angle)<0.01f)
        {
            return i;
        }
    }
    return 0;
}
