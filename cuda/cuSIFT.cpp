#include "cuSIFT.h"


namespace cusift {

void createInitialImage_gpu(const Mat &src, CudaImage &base, float sigma, bool doubleImageSize);

}


namespace cusift {
cuSIFT::cuSIFT():
  nOctaveLayers(3),
  contrastThreshold(0.04),
  edgeThreshold(10),
  sigma(1.6)
{

}

void cuSIFT::detectAndCompute(cv::Mat& src,std::vector<cv::KeyPoint> keypoints){
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;

    CudaImage base;
    createInitialImage_gpu(src,base,sigma,firstOctave<0);

//    int nOctaveLayers = 3;

//    int nOctaves = cvRound(std::log( (double)std::min( base.width, base.height ) ) / std::log(2.) - 2) - firstOctave;

//    std::vector<CudaImage> gpyr,dogpyr;
//    buildGaussianPyramid(base, gpyr, nOctaves);
//    buildDoGPyramid(gpyr, dogpyr);
//    float* h_keypoints;
//    findScaleSpaceExtrema(gpyr, dogpyr, h_keypoints);
}


}
