/*
 * ORB CUDA
 */
//#include <opencv2/imgcodecs.hpp>
//#include <cuda.h>
//#include <iostream>
//#include <opencv2/cudafeatures2d.hpp>
//#include <vector>
//#include "opencv2/cudaarithm.hpp"
//#include "opencv2/cudaimgproc.hpp"

//using namespace cv;
//int main()
//{
//    Mat img_1 = imread("../data/img2.ppm");
//    Mat img_2 = imread("../data/img3.ppm");

//    cuda::GpuMat d_img1, d_img2;
//    cuda::GpuMat d_srcL, d_srcR;

//    d_img1.upload(img_1); d_img2.upload(img_2);

//    Mat img_matches, des_L, des_R;

//    cuda::cvtColor(d_img1, d_srcL, COLOR_BGR2GRAY);
//    cuda::cvtColor(d_img2, d_srcR, COLOR_BGR2GRAY);

//    Ptr<cuda::ORB> d_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20,true);

//    cuda::GpuMat d_keypointsL, d_keypointsR;
//    cuda::GpuMat d_descriptorsL, d_descriptorsR, d_descriptorsL_32F, d_descriptorsR_32F;

//    std::vector<KeyPoint> keyPoints_1, keyPoints_2;

//    Ptr<cv::cuda::DescriptorMatcher> d_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

//    std::vector<DMatch> matches;
//    std::vector<DMatch> good_matches;

//    d_orb -> detectAndComputeAsync(d_srcL, cuda::GpuMat(), d_keypointsL, d_descriptorsL);
//    d_orb -> convert(d_keypointsL, keyPoints_1);
//    d_descriptorsL.convertTo(d_descriptorsL_32F, CV_32F);

//    d_orb -> detectAndComputeAsync(d_srcR, cuda::GpuMat(), d_keypointsR, d_descriptorsR);
//    d_orb -> convert(d_keypointsR, keyPoints_2);
//    d_descriptorsR.convertTo(d_descriptorsR_32F, CV_32F);

//    d_matcher -> match(d_descriptorsL_32F, d_descriptorsR_32F, matches);


//    std::cout<<"Asd!"<<std::endl;

//    return 0;
//}

/*
 * SURF CUDA  and  ORB CUDA
 */
//#include <iostream>

//#include "opencv2/opencv_modules.hpp"

//#ifdef HAVE_OPENCV_XFEATURES2D

//#include "opencv2/core.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/cudafeatures2d.hpp"
//#include "opencv2/xfeatures2d/cuda.hpp"
//#include "opencv2/cudaarithm.hpp"
//#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"


//#include "opencv2/highgui.hpp"
//#include "opencv2/calib3d.hpp"
//using namespace std;
//using namespace cv;
//using namespace cv::cuda;

//static void help()
//{
//    cout << "\nThis program demonstrates using SURF_CUDA features detector, descriptor extractor and BruteForceMatcher_CUDA" << endl;
//    cout << "\nUsage:\n\tsurf_keypoint_matcher --left <image1> --right <image2>" << endl;
//}

//#define USE_CUDA_SURF
////#define USE_CUDA_ORB

//int main(int argc, char* argv[])
//{

//    GpuMat img1, img2;
//    GpuMat imgR1,imgR2;
//    GpuMat keypoints1GPU, keypoints2GPU;
//    GpuMat descriptors1GPU, descriptors2GPU;
//    vector<KeyPoint> keypoints1, keypoints2;
//    vector<float> descriptors1, descriptors2;

//    char * p1 = "/home/jie/workspace/projects/CUDA_SIfT/Qt_cuda_sift/data/100_7101.JPG";
//    char * p2 = "/home/jie/workspace/projects/CUDA_SIfT/Qt_cuda_sift/data/100_7102.JPG";

//    imgR1.upload(imread(p1));
//    imgR2.upload(imread(p2));

//    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

//    cuda::cvtColor(imgR1, img1, COLOR_BGR2GRAY);
//    cuda::cvtColor(imgR2, img2, COLOR_BGR2GRAY);
//#ifdef USE_CUDA_SURF
//    //////////////////////
//    /// CUDA SURF
//    /////////////////////
//    SURF_CUDA surf;

//    // detecting keypoints & computing descriptors

//    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
//    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

//    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
//    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;


//    // downloading results

//    surf.downloadKeypoints(keypoints1GPU, keypoints1);
//    surf.downloadKeypoints(keypoints2GPU, keypoints2);
//    surf.downloadDescriptors(descriptors1GPU, descriptors1);
//    surf.downloadDescriptors(descriptors2GPU, descriptors2);

//#endif
//#ifdef USE_CUDA_ORB
//    /////////////////////////
//    /// CUDA ORB
//    /////////////////////////

//    Ptr<cuda::ORB> d_orb = cuda::ORB::create();
//    cuda::GpuMat d_descriptorsL, d_descriptorsR;
//    d_orb -> detectAndComputeAsync(img1, cuda::GpuMat(), keypoints1GPU, d_descriptorsL);
//    d_orb -> convert(keypoints1GPU, keypoints1);
//    d_descriptorsL.convertTo(descriptors1GPU, CV_32F);

//    d_orb -> detectAndComputeAsync(img2, cuda::GpuMat(), keypoints2GPU, d_descriptorsR);
//    d_orb -> convert(keypoints2GPU, keypoints2);
//    d_descriptorsR.convertTo(descriptors2GPU, CV_32F);

//#endif

//    // matching descriptors
//    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
//    vector<DMatch> matches;
//    matcher->match(descriptors1GPU, descriptors2GPU, matches);


//    //-- Quick calculation of max and min distances between keypoints
//    double max_dist = 0; double min_dist = 100;
//    for( int i = 0; i < descriptors1GPU.rows; i++ )
//    { double dist = matches[i].distance;
//      if( dist < min_dist ) min_dist = dist;
//      if( dist > max_dist ) max_dist = dist;
//    }
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
//    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//    std::vector< DMatch > good_matches;
//    for( int i = 0; i < descriptors1GPU.rows; i++ )
//    { if( matches[i].distance <= 3*min_dist )
//       { good_matches.push_back( matches[i]); }
//    }
//    std::cout<<"good_matches num:"<<good_matches.size()
//            <<" <= 3*min_dist rate:"<<(float)good_matches.size()/matches.size();


//    // drawing the results
//    Mat img_matches;
//    drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, good_matches, img_matches,
//                Scalar::all(-1),Scalar(0,0,255),std::vector<char>(),0);

//    namedWindow("matches", 0);
//    resizeWindow("matches",Size(1600,1600));
//    imshow("matches", img_matches);
//    waitKey(0);

//    //-- Localize the object
//    std::vector<Point2f> obj;
//    std::vector<Point2f> scene;
//    for( size_t i = 0; i < good_matches.size(); i++ )
//    {
//      //-- Get the keypoints from the good matches
//      obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
//      scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
//    }
//    Mat H = findHomography( obj, scene, RANSAC );
//    for(int i = 0;i < H.cols;i++){
//        for(int j = 0;j<H.rows;j++)
//            std::cout<<H.at<double>(i,j)<< "  ";
//            //std::cout<<((double *)H.data)[i*H.cols+j]<< "  ";
//            //std::cout<<H.type()<< "  ";
//        std::cout<<std::endl;
//    }

//    //-- Get the corners from the image_1 ( the object to be "detected" )
//    std::vector<Point2f> obj_corners(4);
//    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
//    obj_corners[2] = cvPoint( img1.cols, img2.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
//    std::vector<Point2f> scene_corners(4);
//    perspectiveTransform( obj_corners, scene_corners, H);
//    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
//    //-- Show detected matches
//    cvNamedWindow("Good Matches & Object detection",CV_WINDOW_NORMAL);
//    imshow( "Good Matches & Object detection", img_matches );
//    waitKey(0);

//    return 0;
//}

//#else

//int main()
//{
//    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
//    return 0;
//}

//#endif



/*
 *
 * test my API
 *
 *
 */
#include "cuda/cuSIFT.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda/cuGlobal.h"
#include "cuda/cuImage.h"
#include "cuda/cudaImage.h"
#include "cuda/cusitf_function_H.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>
#include "sift/sift_cv_gpu.h"
#define USE_MY_SIFT


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
    cv::cuda::GpuMat src_gpu;
    src_gpu.upload(cv::imread("../data/100_7101.JPG",cv::IMREAD_GRAYSCALE));
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
//    Mat kepoint;
//    drawKeypoints(src, keypoints,kepoint,cv::Scalar::all(-1),4);
//    cvNamedWindow("old cuda sift",CV_WINDOW_NORMAL);
//    imshow("old cuda sift", kepoint);
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
    keypointss.resize(28000);
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
//    Mat kepoint2;
//    Mat dst(src_gpu),img;
//    dst.convertTo(img, DataType<uchar>::type, 1, 0);
//    drawKeypoints(img, keypointss,kepoint2,cv::Scalar::all(-1),4);
//    cvNamedWindow("new cuda sift",CV_WINDOW_NORMAL);
//    imshow("new cuda sift", kepoint2);
    //等待任意按键按下
    //waitKey(0);

//    Mat descriptors_show(descriptsGPU);
//    Mat ss;
//    descriptors_show.convertTo(ss, DataType<uchar>::type, 1, 0);
//    cvNamedWindow("new descriptors",CV_WINDOW_NORMAL);
//    imshow("new descriptors", ss);
    //////////////////////////////////////////////////





    ///////////////////////////////
    /// original sift
    ///////////////////////////////
//    //Create SIFT class pointer
//#ifdef USE_MY_SIFT
//    Ptr<Feature2D> f2d = xfeatures2d::cvGpu::SIFT::create();
//#else
//    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//#endif
//    //读入图片
//    Mat img_1 = src;

//    //Detect the keypoints
//    std::vector<KeyPoint> keypoints_1, keypoints_2;

//#ifdef TIME
//    t = (double)getTickCount();
//#endif
//    f2d->detect(img_1, keypoints_1);

//    //Calculate descriptors (feature vectors)
//    Mat descriptors_1, descriptors_2;
//    f2d->compute(img_1, keypoints_1, descriptors_1);

//#ifdef TIME
//    t = (double)getTickCount() - t;
//    printf("opencv sift cost : %g ms\n", t*1000./tf);//158
//#endif

//    std::cout<<"Original sift keypoints num :"<<keypoints_1.size()<<std::endl;
//    Mat kepoint1;
//    drawKeypoints(src, keypoints_1,kepoint1,cv::Scalar::all(-1),4);
//    cvNamedWindow("extract_cpu",CV_WINDOW_NORMAL);
//    imshow("extract_cpu", kepoint1);
//    //等待任意按键按下
//    waitKey(0);
    //////////////////////////////////////////////////




    return 0;
}
