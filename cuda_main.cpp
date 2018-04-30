#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>

#define USE_MY_SIFTs


#ifdef USE_SIFT OR USE_SURF
#include "opencv2/features2d.hpp"
#endif
#ifdef USE_MY_SIFT
#include"sift/sift.h"
#endif



//#define IMAGE_SHOW
using namespace cv;
using namespace std;



#define TIME


int main()
{
    std::cout<<"Hello World !"<<std::endl;

    //char *a ="../data/road.png";
    //char *a ="../data/lena.png";
    //char *a ="../data/100_7101.JPG";
    //char *a ="../data/DSC04034.JPG";
    char *a ="../data/1080.jpg";
    cv::Mat src;
    //src = imread("../data/100_7101.JPG",0);
    //src = imread("../data/lena.png",0);
    src = imread(a);
    //src = imread("../data/DSC04034.JPG",0);
    int width = src.cols;
    int height = src.rows;
    if(!src.data)
    {
        printf("no photo");
    }
    int sigma1 = 1.6;

    Mat tmp;
    src.convertTo(tmp, CV_32FC1);

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

    cuimg.Download();
//    cuGaussianBlur(cuimg,sigma1);


    //nodoublesize
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
#ifdef IMAGE_SHOW
    Mat dis(base.height,base.width,CV_32F);
    safeCall(cudaMemcpy2D(dis.data,base.width*sizeof(float),base.d_data,base.pitch*sizeof(float),base.width*sizeof(float),(size_t) base.height,cudaMemcpyDeviceToHost));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 0);
    cvNamedWindow("ss",CV_WINDOW_NORMAL);
    imshow("ss",gray);
    waitKey(0);
#endif

    int nOctaveLayers = 3;
#ifdef TEST_FIRST_OCTAVE
    int nOctaves = cvRound(std::log( (double)std::min( cuimg.width, cuimg.height ) ) / std::log(2.) - 8) - firstOctave;
#else
    int nOctaves = cvRound(std::log( (double)std::min( cuimg.width, cuimg.height ) ) / std::log(2.) - 2) - firstOctave;
#endif
    std::vector<CudaImage> gpyr,dogpyr;
    buildGaussianPyramid(base, gpyr, nOctaves);
    buildDoGPyramid(gpyr, dogpyr);

    float* keypoints;

    findScaleSpaceExtrema(gpyr, dogpyr, keypoints);


//    std::vector<double> sig(nOctaveLayers + 3);
//    //init the size of the pyramid images which is nOctave*nLayer

//    float sigma = 1.6;
//    sig[0] = sigma;
//    double k = std::pow( 2., 1. / nOctaveLayers );
//    for( int i = 1; i < nOctaveLayers + 3; i++ )
//    {
//        double sig_prev = std::pow(k, (double)(i-1))*sigma;
//        double sig_total = sig_prev*k;
//        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
//    }

    //computePerOctave(cuimg,sig,nOctaveLayers);
//    float *d_data;
//    safeCall(cudaMalloc(&d_data,1<<30));


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
    Mat img_1 = imread(a);

    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);

    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);

    for(int i = 0;i<keypoints_1.size();++i){
//        keypoints_1[i].angle = 0;
//        keypoints_1[i].class_id = 0;
//        keypoints_1[i].octave = 0;
//        keypoints_1[i].response = 0;
//        keypoints_1[i].size = 0;
        //keypoints_1[i].pt.x = 0;
    }
    std::cout<<"sift keypoints num :"<<keypoints_1.size()<<std::endl;
    Mat kepoint;
    drawKeypoints(img_1, keypoints_1,kepoint,cv::Scalar::all(-1),4);
    cvNamedWindow("extract",CV_WINDOW_NORMAL);
    imshow("extract", kepoint);
    //等待任意按键按下
    waitKey(0);


#ifdef COMPARE_VALUE
//    vector<KeyPoint> k;
//    vector<KeyPoint> b;
//    std::sort(k.begin(),k.end(),comparex);
    sort(keypoints_1.begin(),keypoints_1.end(),sortx);
    int unique_nums;
    unique_nums = std::unique(keypoints_1.begin(),keypoints_1.end(),uniquex) - keypoints_1.begin();
    for(int i = 0;i < unique_nums;i++)
        std::cout<<keypoints_1[i].response<<" ";
    std::cout<<unique_nums<<std::endl;

#endif



    return 0;
}




//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/xfeatures2d/cuda.hpp>
//#include <opencv2/cudafeatures2d.hpp>

//using namespace std;

//int GetMatchPointCount(const char * pic_path_1,const char * pic_path_2) {
//  /*指定使用的GPU序号，相关的还有下面几个函数可以使用
//    cv::cuda::getCudaEnabledDeviceCount();
//    cv::cuda::getDevice();
//    cv::cuda::DeviceInfo*/
//  cv::cuda::setDevice(0);

//  /*向显存加载两张图片。这里需要注意两个问题：
//    第一，我们不能像操作（主）内存一样直接一个字节一个字节的操作显存，也不能直接从外存把图片加载到显存，一般需要通过内存作为媒介
//    第二，目前opencv的GPU SURF仅支持8位单通道图像，所以加上参数IMREAD_GRAYSCALE*/
//  cv::cuda::GpuMat gmat1;
//  cv::cuda::GpuMat gmat2;
//  gmat1.upload(cv::imread(pic_path_1,cv::IMREAD_GRAYSCALE));
//  gmat2.upload(cv::imread(pic_path_2,cv::IMREAD_GRAYSCALE));

//  /*下面这个函数的原型是：
//  explicit SURF_CUDA(double
//      _hessianThreshold, //SURF海森特征点阈值
//      int _nOctaves=4, //尺度金字塔个数
//      int _nOctaveLayers=2, //每一个尺度金字塔层数
//      bool _extended=false, //如果true那么得到的描述子是128维，否则是64维
//      float _keypointsRatio=0.01f,
//      bool _upright = false
//      );
//  要理解这几个参数涉及SURF的原理*/
//  cv::cuda::SURF_CUDA surf(
//      100,4,3
//      );

//  /*分配下面几个GpuMat存储keypoint和相应的descriptor*/
//  cv::cuda::GpuMat keypt1,keypt2;
//  cv::cuda::GpuMat desc1,desc2;

//  /*检测特征点*/
//  surf(gmat1,cv::cuda::GpuMat(),keypt1,desc1);
//  surf(gmat2,cv::cuda::GpuMat(),keypt2,desc2);

//  /*匹配，下面的匹配部分和CPU的match没有太多区别,这里新建一个Brute-Force Matcher，一对descriptor的L2距离小于0.1则认为匹配*/
//  auto matcher=cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
//  vector<cv::DMatch> match_vec;
//  matcher->match(desc1,desc2,match_vec);

//  int count=0;
//  for(auto & d:match_vec){
//    if(d.distance<0.1) count++;
//  }
//  return count;
//}

//int main(int argc, const char* argv[])
//{
//  char * p1 = "/home/jie/workspace/projects/CUDA_SIfT/Qt_cuda_sift/data/100_7100.JPG";
//  char * p2 = "/home/jie/workspace/projects/CUDA_SIfT/Qt_cuda_sift/data/100_7101.JPG";
//  GetMatchPointCount(p1,p2);
//  return 0;
//}
