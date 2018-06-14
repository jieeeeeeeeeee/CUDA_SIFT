
//#include "cuImage.h"
//namespace cusift {

//void createInitialImage_gpu(const Mat &src, CudaImage &base, float sigma, bool doubleImageSize);

//}


//namespace cusift {
//cuSIFT::cuSIFT():
//  nOctaveLayers(3),
//  contrastThreshold(0.04),
//  edgeThreshold(10),
//  sigma(1.6)
//{

//}

//void cuSIFT::detectAndCompute(cv::Mat& src,std::vector<cv::KeyPoint> keypoints){
//    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;

//    CudaImage base;
//    createInitialImage_gpu(src,base,sigma,firstOctave<0);

////    int nOctaveLayers = 3;

////    int nOctaves = cvRound(std::log( (double)std::min( base.width, base.height ) ) / std::log(2.) - 2) - firstOctave;

////    std::vector<CudaImage> gpyr,dogpyr;
////    buildGaussianPyramid(base, gpyr, nOctaves);
////    buildDoGPyramid(gpyr, dogpyr);
////    float* h_keypoints;
////    findScaleSpaceExtrema(gpyr, dogpyr, h_keypoints);
//}


//}


#include "cuSIFT.h"
#include "cuSIFT_D.cu"

namespace cv {
namespace cuda {
namespace device {
namespace sift {

/******************************* Defs and macros *****************************/

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


void differenceImg_gpu(const PtrStepf& next,const PtrStepf& prev,const PtrStepf& diff)
{
//    static int num = 0;
//    differenceImg_gpu<<<1,1>>>();
//    std::cout<<"static function "<<num++<<std::endl;
//    diff.create(next.rows,next.cols,next.type());
//    dim3 Block(32,8);
//    dim3 Grid(iDivUp(next.step1(),Block.x),iDivUp(next.rows,Block.y));
    //differenceImg_gpu1<<<1,1>>>(next,prev,diff);

}




void createInitialImage(const GpuMat& src,GpuMat& base, bool doubleImageSize, float sigma)
{
    int width = src.cols;
    int height = src.rows;

    //convert the input Mat type is CV_8U to CV_32F for the calculate below
    GpuMat gray_fpt;
    src.convertTo(gray_fpt, DataType<float>::type, 1, 0);
    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        cuda::resize(gray_fpt, gray_fpt, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
        width = gray_fpt.cols;
        height = gray_fpt.rows;
        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(CV_32F, CV_32F, Size(0, 0), sig_diff, 0, cv::BORDER_DEFAULT,-1);
        gauss->apply(gray_fpt,base);
//            GpuMat test;
//            gray_fpt.convertTo(test, DataType<uchar>::type, 1, 0);
//            cv::Mat show(test);
//            cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//            cv::imshow("show",show);
//            cv::waitKey(0);
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(CV_32F, CV_32F, Size(0, 0), sig_diff, 0, cv::BORDER_DEFAULT,-1);
        gauss->apply(gray_fpt,base);
//            GpuMat test;
//            base.convertTo(test, DataType<uchar>::type, 1, 0);
//            cv::Mat show(test);
//            cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//            cv::imshow("show",show);
//            cv::waitKey(0);
    //GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
    }
}

void buildGaussianPyramid(GpuMat& base, std::vector<GpuMat>& pyr, int nOctaves,int nOctaveLayers,float sigma)
{
    //the vector of sigma per octave
    std::vector<double> sig(nOctaveLayers + 3);
    //init the size of the pyramid images which is nOctave*nLayer
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            GpuMat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const GpuMat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                cuda::resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);
            }
            else
            {
                const GpuMat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(CV_32F, CV_32F, Size(0, 0), sig[i], 0, cv::BORDER_DEFAULT,-1);
                gauss->apply(src,dst);
//                    GpuMat test;
//                    dst.convertTo(test, DataType<uchar>::type, 1, 0);
//                    cv::Mat show(test);
//                    cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//                    cv::imshow("show",show);
//                    cv::waitKey(0);
            }
        }
    }
}

void buildDoGPyramid(std::vector<GpuMat>& gpyr, std::vector<GpuMat>& dogpyr,int nOctaveLayers)
{
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );

    //could use cuda stream
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers + 2;i++){
            GpuMat& prev = gpyr[o*(nOctaveLayers + 3)+i];
            GpuMat& next = gpyr[o*(nOctaveLayers + 3)+i+1];
            GpuMat& diff = dogpyr[o*(nOctaveLayers + 2)+i];
//                diff.Allocate(prev.width,prev.height,prev.pitch,false);
//                dim3 Block(32,8);
//                dim3 Grid(iDivUp(diff.pitch,Block.x),iDivUp(diff.height,Block.y));
//                differenceImg<<<Grid,Block>>>(prev.d_data,next.d_data,diff.d_data,diff.pitch,diff.height);
//                safeCall(cudaDeviceSynchronize());
#if 0
            cuda::subtract(next, prev, diff, noArray(), DataType<float>::type);
#else
            differenceImg_gpu(next,prev,diff);
#endif
//                GpuMat test;
//                diff.convertTo(test, DataType<uchar>::type, 100, 0);
//                cv::Mat show(test);
//                cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//                cv::imshow("show",show);
//                cv::waitKey(0);
        }
    }
}
}
}
}
}


using namespace cv::cuda;
using namespace ::cv::cuda::device::sift;
namespace {
class SIFT_CUDA_Invoker{
public:
    SIFT_CUDA_Invoker(cv::cuda::SIFT_CUDA& _sift, const GpuMat& img, const GpuMat& mask) :
        sift(_sift),
        img_cols(img.cols), img_rows(img.rows),
        use_mask(!mask.empty())
    {
        CV_Assert(!img.empty() && img.type() == CV_8UC1);
        CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));



        if (use_mask)
        {

        }
    }

    void detectKeypoints(GpuMat& keypoints)
    {

    }

    void detectAndCompute(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors,
                                bool useProvidedKeypoints)
    {
        int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
        if( img.empty() || img.depth() != CV_8U )
            CV_Error( cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

        if( !mask.empty() && mask.type() != CV_8UC1 )
            CV_Error( cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

        //no need for detection using for desription.
        if( useProvidedKeypoints )
        {
//            firstOctave = 0;
//            int maxOctave = INT_MIN;
//            for( size_t i = 0; i < keypoints.size(); i++ )
//            {
//                int octave, layer;
//                float scale;
//                unpackOctave(keypoints[i], octave, layer, scale);
//                firstOctave = std::min(firstOctave, octave);
//                maxOctave = std::max(maxOctave, octave);
//                actualNLayers = std::max(actualNLayers, layer-2);
//            }

//            firstOctave = std::min(firstOctave, 0);
//            CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
//            actualNOctaves = maxOctave - firstOctave + 1;
        }

        //create the zero(or -1) Octave image `base`
        cv::cuda::GpuMat base;
        //output base Mat
        createInitialImage(img,base,firstOctave < 0, sift.sigma);


        //the number of the Octaves which can calculate by formula "|log2 min(X,Y) - 2|"
        //cvRound Rounds floating-point number to the nearest integer.
        int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - firstOctave;

        std::vector<GpuMat> gpyr,dogpyr;
        //double t, tf = getTickFrequency();
        //t = (double)getTickCount();
        buildGaussianPyramid(base, gpyr, nOctaves,sift.nOctaveLayers,sift.sigma);
        buildDoGPyramid(gpyr, dogpyr,sift.nOctaveLayers);

        //t = (double)getTickCount() - t;
        //printf("pyramid construction time: %g\n", t*1000./tf);

//        if( !useProvidedKeypoints )
//        {
//            //t = (double)getTickCount();
//            findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
//            KeyPointsFilter::removeDuplicatedSorted( keypoints );

//            if( nfeatures > 0 )
//                KeyPointsFilter::retainBest(keypoints, nfeatures);
//            //t = (double)getTickCount() - t;
//            //printf("keypoint detection time: %g\n", t*1000./tf);


//            if( firstOctave < 0 )
//                for( size_t i = 0; i < keypoints.size(); i++ )
//                {
//                    KeyPoint& kpt = keypoints[i];
//                    float scale = 1.f/(float)(1 << -firstOctave);
//                    kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
//                    kpt.pt *= scale;
//                    kpt.size *= scale;
//                }

//            if( !mask.empty() )
//                KeyPointsFilter::runByPixelsMask( keypoints, mask );
//        }
//        else
//        {
//            // filter keypoints by mask
//            //KeyPointsFilter::runByPixelsMask( keypoints, mask );
//        }

//        if( _descriptors.needed() )
//        {
//            //t = (double)getTickCount();
//            int dsize = descriptorSize();
//            _descriptors.create((int)keypoints.size(), dsize, CV_32F);
//            Mat descriptors = _descriptors.getMat();

//            calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
//            //t = (double)getTickCount() - t;
//            //printf("descriptor extraction time: %g\n", t*1000./tf);
//        }
    }

private:
    cv::cuda::SIFT_CUDA& sift;

    int img_cols, img_rows;

    bool use_mask;
};
}


/*
 *  SIFT_CUDA is a API for host and device
 *  SIFT_CUDA_Invoker is the API for compute in device
 *  SIFT_CUDA_Invoker() is a initialize for device environment
 *  SIFT_CUDA_Invoker has the function for lanuch the kernel
 *  The cuSIFT.cu is all of the kernel functions.
 */

namespace cv {
namespace cuda {
//    SIFT_CUDA::SIFT_CUDA()
//    {
//        nOctaveLayers = 3;
//        contrastThreshold = 0.04;
//        edgeThreshold = 10;
//        sigma = 1.6;
//    }
    SIFT_CUDA::SIFT_CUDA(int _nOctaveLayers,
                         float _contrastThreshold, float _edgeThreshold,
                         float _sigma)
    {
        nOctaveLayers = _nOctaveLayers;
        contrastThreshold = _contrastThreshold;
        edgeThreshold = _edgeThreshold;
        sigma = _sigma;
    }
    void SIFT_CUDA::operator ()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors,
                                bool useProvidedKeypoints)
    {

        if (!img.empty())
        {

            SIFT_CUDA_Invoker sift(*this, img, mask);

            sift.detectAndCompute(img,mask,keypoints,descriptors,useProvidedKeypoints);
            //if (!useProvidedKeypoints)
                //sift.detectKeypoints(keypoints);

            //sift.computeDescriptors(keypoints, descriptors, descriptorSize());
        }


    }




}

}
