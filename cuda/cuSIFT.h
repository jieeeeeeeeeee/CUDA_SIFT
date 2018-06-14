#ifndef CUSIFT_H
#define CUSIFT_H

//#include "cuImage.h"



//namespace cusift {

//class cuSIFT{
//public:
//    cuSIFT();
//    void detectAndCompute(Mat &src, std::vector<KeyPoint> keypoints);
//private:
//    int nOctaveLayers;
//    double contrastThreshold;
//    double edgeThreshold;
//    double sigma;

//};

//}

#include"cuGlobal.h"


namespace cv{

namespace cuda {

/*
 *
 * SIFT_CUDA is a API that has create or () initiazation,detect using (),compute using(),
 * download keypoints descripters,
 * SIFT_CUDA_Invorkers has the detectAndcompute buildpyramid and detectpoints
 * and calOritend
 *
 */


class SIFT_CUDA{
public:
    //! the default constructor
    //SIFT_CUDA();
    //! the full constructor taking all the necessary parameters
    SIFT_CUDA(int _nOctaveLayers = 3,
                   float _contrastThreshold = 0.04, float _edgeThreshold = 10,
                   float _sigma = 1.6);

    //! returns the descriptor size in floats (128)
    int descriptorSize() const;

    //! returns the descriptor type
    int descriptorType() const;

    //! returns the default norm type
    int defaultNorm() const;

    //! upload host keypoints to device memory
    void uploadKeypoints(const std::vector<KeyPoint>& keypoints, GpuMat& keypointsGPU);

    //! download keypoints from device to host memory
    void downloadKeypoints(const GpuMat& keypointsGPU, std::vector<KeyPoint>& keypoints);

    //! download descriptors from device to host memory
    void downloadDescriptors(const GpuMat& descriptorsGPU, std::vector<float>& descriptors);

    //! finds the keypoints using fast hessian detector used in SURF
    //! supports CV_8UC1 images
    //! keypoints will have nFeature cols and 6 rows
    //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
    //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
    //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
    //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
    //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
    //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
    //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
    void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints);
    //! finds the keypoints and computes their descriptors.
    //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
    void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors,
        bool useProvidedKeypoints = false);

    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints);
    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors,
        bool useProvidedKeypoints = false);

    void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, std::vector<float>& descriptors,
        bool useProvidedKeypoints = false);




    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
};

}

}




#endif
