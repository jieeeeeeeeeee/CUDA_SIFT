#include "cuImage.h"

namespace cusift {
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

static const int SIFT_FIXPT_SCALE = 1;

}

namespace cusift {

void createInitialImage_gpu(const Mat &src, CudaImage &base, float sigma, bool doubleImageSize){
    int width = src.cols;
    int height = src.rows;
    if(!src.data){
        printf("input none data !");
        return;
    }

    Mat gray, gray_fpt;
    if( src.channels() == 3 || src.channels() == 4 )
    {
        cvtColor(src, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, DataType<float>::type, 1, 0);
    }
    else
        src.convertTo(gray_fpt, DataType<float>::type, 1, 0);

    //sigma different which is sqrt(1.6*1.6-0.5*0.5*4)
    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        resize(gray_fpt, gray_fpt, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
        width = gray_fpt.cols;
        height = gray_fpt.rows;
        base.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)gray_fpt.data);
        base.Download();
        //cuGaussianBlur(base,sig_diff);

    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        base.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)gray_fpt.data);
        base.Download();
        //cuGaussianBlur(base,sig_diff);
        //GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
    }

}

}
