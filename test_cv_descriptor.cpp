#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>
#include "cuda/cuSIFT.h"

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
int findSamePointsIndex(cv::KeyPoint& keypoint,std::vector<cv::KeyPoint>&keypoints);
static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
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
    cv::cuda::GpuMat src_gpu;
    src_gpu.upload(cv::imread("../data/road.png",cv::IMREAD_GRAYSCALE));
    ///////////////////////////////
    /// old cuda sift
    ///////////////////////////////
    Mat src(src_gpu);
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
    printf("old cuda sift cost : %g ms\n", t*1000./tf);//246ms
#endif

    //std::cout<<"sift keypoints num :"<<keypoints.size()<<std::endl;
    Mat kepoint;
    drawKeypoints(src, keypoints,kepoint,cv::Scalar::all(-1),4);
    cvNamedWindow("old cuda sift",CV_WINDOW_NORMAL);
    imshow("old cuda sift", kepoint);
    //等待任意按键按下
    //waitKey(0);
    //////////////////////////////////////////////////

#ifdef TIME

    t = (double)getTickCount();
#endif

    ////////////////////////////////
    /// new cuda sift
    ////////////////////////////////
//    cv::namedWindow("show");
//    cv::imshow("show",cv::Mat(src));
//    cv::waitKey(0);
    cv::cuda::GpuMat keypointsGPU,descriptsGPU;
    cv::cuda::SIFT_CUDA sift;
    sift(src_gpu,cv::cuda::GpuMat(),keypointsGPU,descriptsGPU);

//    Ptr<cuda::ORB> d_orb = cuda::ORB::create();

//    cv::cuda::SURF_CUDA surf;

    // detecting keypoints & computing descriptors

    //surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);


    Mat keypointsCPU(keypointsGPU);
    float* h_keypoints = (float*)keypointsCPU.ptr();
    std::vector<cv::KeyPoint>keypointss;
    keypointss.resize(1000);
    for(int i = 0;i<keypointss.size();++i)
    {
        keypointss[i].pt.x =  h_keypoints[i];
        keypointss[i].pt.y =  h_keypoints[i+keypointsCPU.step1()*1];
        keypointss[i].octave =  h_keypoints[i+keypointsCPU.step1()*2];
        keypointss[i].size =  h_keypoints[i+keypointsCPU.step1()*3];
        keypointss[i].response =  h_keypoints[i+keypointsCPU.step1()*4];
        keypointss[i].angle =  h_keypoints[i+keypointsCPU.step1()*5];
    }
    int firstOctave = -1;
    if( firstOctave < 0 )
        for( size_t i = 0; i < keypointss.size(); i++ )
        {
            KeyPoint& kpt = keypointss[i];
            float scale = 1.f/(float)(1 << -firstOctave);
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }
#ifdef TIME
    t = (double)getTickCount() - t;
    printf("new cuda sift cost : %g ms\n", t*1000./tf);//246ms
#endif
    Mat kepoint2;
    Mat dst(src_gpu),img;
    dst.convertTo(img, DataType<uchar>::type, 1, 0);
    drawKeypoints(img, keypointss,kepoint2,cv::Scalar::all(-1),4);
    cvNamedWindow("new cuda sift",CV_WINDOW_NORMAL);
    imshow("new cuda sift", kepoint2);
    //等待任意按键按下
    //waitKey(0);

    Mat descriptors_show(descriptsGPU);
    Mat ss;
    descriptors_show.convertTo(ss, DataType<uchar>::type, 1, 0);
    cvNamedWindow("new descriptors",CV_WINDOW_NORMAL);
    imshow("new descriptors", ss);
    //////////////////////////////////////////////////

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
    std::cout<<"sift keypoints num :"<<keypoints_1.size()<<std::endl;
    //Mat kepoint;
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


#define TEST_DESCRIPTOR
#ifdef TEST_DESCRIPTOR


    int difcount=0;
    int k = 0;
    std::map<int,int> map;
    for(int i = 0;i<keypoints_1.size();i++)
    {
        int idx = findSamePointsIndex(keypoints_1[i],keypointss);

        if(idx){
            //printf("%d -- %d \n",i,idx);
            map.insert(std::pair<int,int>(i,idx));
            k++;
        }
    }
    //printf("k: %d -- %d \n",k,(int)keypoints_1.size());
    if(keypoints_1.size()==k)
        printf("all match ! \n");
    else
        printf("not all match ! match rate : %f\n",(float)k/keypoints_1.size());

    cv::Mat difImg;
    difImg.create(k,128,CV_8UC1);
    memset(difImg.data,0,difImg.cols*difImg.rows*sizeof(uchar));


    std::map<int,int>::iterator iter;
    int i = 0;
    for(iter = map.begin();iter!=map.end();iter++)
    {
        float* psift = descriptors_1.ptr<float>(iter->first);
        float* pcuda = descriptors_show.ptr<float>(iter->second);
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
                difcount++;
                break;
            }
        }
        i++;
    }

    std::cout<<"same descriptor rate:"<<float(keypoints_1.size()-difcount)/keypoints_1.size()<<std::endl;

    cvNamedWindow("dif",CV_WINDOW_NORMAL);
    imshow("dif", difImg);
    waitKey(0);



#endif



    return 0;
}
int findSamePointsIndex(cv::KeyPoint& keypoint,std::vector<cv::KeyPoint>&keypoints){

    for(int i = 0;i<keypoints.size();i++)
    {
        if(abs(keypoint.pt.x-keypoints[i].pt.x)<=0.001&&abs(keypoint.pt.y-keypoints[i].pt.y)<=0.001 && std::abs(keypoint.angle-keypoints[i].angle)<0.01f)
        {
            return i;
        }
    }
    return 0;
}
