#ifndef CUSIFT_FUNCTION_H_H
#define CUSIFT_FUNCTION_H_H

#include "cuImage.h"
#include "cuGlobal.h"
#include "cudaImage.h"
//#include "cusitf_function_D.h"

using namespace cv;



#define SHOW
//#define SHOW_GAUSSIANPYRAMID
//#define SHOW_DOGPYRAMID
#define SHOW_KEYPOINT

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

//the value of static value
int nOctaveLayers = 3;
double contrastThreshold;
double edgeThreshold;
double sigma = 1.6;
unsigned int maxPoints = 2000;


cv::Mat cv::getGaussianKernel( int n, double sigma, int ktype );

void disMatf(CudaImage &cuImg);


extern "C"
void cuGaussianBlur(CudaImage& cuImg,float sigma);

extern "C"
void buildPyramidNoStream(const CudaImage& base, std::vector<CudaImage>& pyr, int nOctaves ,int nOctaveLayers );

extern "C"
void computePerOctave(CudaImage& base,std::vector<double> & sig,int nOctaveLayers);

extern "C"
void createInitialImage(const Mat &src, CudaImage &base, float sigma, bool doubleImageSize);

extern "C"
void buildGaussianPyramid(CudaImage& base, std::vector<CudaImage>& pyr, int nOctaves);

extern "C"
double ScaleDown(CudaImage &res, CudaImage &src, float variance);

extern "C"
void buildDoGPyramid(std::vector<CudaImage>& gpyr, std::vector<CudaImage>& dogpyr );

extern "C"
void findScaleSpaceExtrema(std::vector<CudaImage>& gpyr, std::vector<CudaImage>& dogpyr, float* keypoints);

#endif
