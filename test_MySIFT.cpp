//#include <iostream>
//#include "sift/sift.h"

//#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>
//using namespace cv;  //包含cv命名空间
//using namespace std;

//#define TIME 0



//int main()
//{

//#if TIME
//    double t, tf = getTickFrequency();
//    t = (double)getTickCount();
//#endif


//    //Create SIFT class pointer
//    Ptr<Feature2D> f2d = xfeatures2d::q::SIFT::create();
//    //读入图片
//    Mat img_1 = imread("../data/img2.ppm");
//    Mat img_2 = imread("../data/img3.ppm");
//    //Detect the keypoints
//    vector<KeyPoint> keypoints_1, keypoints_2;
//    f2d->detect(img_1, keypoints_1);
//    f2d->detect(img_2, keypoints_2);
//    //Calculate descriptors (feature vectors)
//    Mat descriptors_1, descriptors_2;
//    f2d->compute(img_1, keypoints_1, descriptors_1);
//    f2d->compute(img_2, keypoints_2, descriptors_2);
//    //Matching descriptor vector using BFMatcher
//    BFMatcher matcher;
//    vector<DMatch> matches;
//    matcher.match(descriptors_1, descriptors_2, matches);
//    //绘制匹配出的关键点
//    Mat img_matches;
//    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
//    cvNamedWindow("match",CV_WINDOW_NORMAL);
//    imshow("match", img_matches);
//    //等待任意按键按下



//#if TIME
//    t = (double)getTickCount() - t;
//    printf("time cost: %g\n", t*1000./tf);
//#endif

//    waitKey(0);
//}


//#define USE_SURF
//#define USE_SIFT
#define USE_MY_SIFT


#ifdef USE_SIFT OR USE_SURF
#include "opencv2/features2d.hpp"
#endif
#ifdef USE_MY_SIFT
#include"sift/sift.h"
#endif
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"


#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
void readme();
/* @function main */



int main( int argc, char** argv )
{
    Mat img_object = imread("/home/jie/workspace/data/Features_Repeatability/vgg_oxford_feat_eval/bark/img1.ppm", IMREAD_GRAYSCALE );
    Mat img_scene  = imread("/home/jie/workspace/data/Features_Repeatability/vgg_oxford_feat_eval/bark/img2.ppm", IMREAD_GRAYSCALE );
    if( !img_object.data || !img_scene.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
    //////////////////////
    /// SURF
    /////////////////////
#ifdef USE_SURF
    //-- Step 1: Detect the keypoints and extract descriptors using SURF
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
#endif
#ifdef USE_SIFT
    /////////////////////////
    /// Original SIFT
    /////////////////////////
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //Detect the keypoints
    vector<KeyPoint> keypoints_object, keypoints_scene;
    f2d->detect(img_object, keypoints_object);
    f2d->detect(img_scene, keypoints_scene);
    //Calculate descriptors (feature vectors)
    Mat descriptors_object, descriptors_scene;
    f2d->compute(img_object, keypoints_object, descriptors_object);
    f2d->compute(img_scene, keypoints_scene, descriptors_scene);
#endif
#ifdef USE_MY_SIFT
    /////////////////////////
    /// MY_SIFT
    /////////////////////////
    Ptr<Feature2D> f2d = xfeatures2d::q::SIFT::create();
    //Detect the keypoints
    vector<KeyPoint> keypoints_object, keypoints_scene;
    f2d->detect(img_object, keypoints_object);
    f2d->detect(img_scene, keypoints_scene);
    //Calculate descriptors (feature vectors)
    Mat descriptors_object, descriptors_scene;
    f2d->compute(img_object, keypoints_object, descriptors_object);
    f2d->compute(img_scene, keypoints_scene, descriptors_scene);
#endif
    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance <= 3*min_dist )
       { good_matches.push_back( matches[i]); }
    }
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    for(int i = 0;i < H.cols;i++){
        for(int j = 0;j<H.rows;j++)
            std::cout<<H.at<double>(i,j)<< "  ";
            //std::cout<<((double *)H.data)[i*H.cols+j]<< "  ";
            //std::cout<<H.type()<< "  ";
        std::cout<<std::endl;
    }

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    cvNamedWindow("Good Matches & Object detection",CV_WINDOW_NORMAL);
    imshow( "Good Matches & Object detection", img_matches );
    waitKey(0);
    return 0;
}
/* @function readme */
void readme()
{ std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
