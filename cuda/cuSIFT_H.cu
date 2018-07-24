
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
#include <opencv2/cudev.hpp>
#include "cuSIFT_D.cu"

namespace cv {
namespace cuda {
namespace device {
namespace sift {


void differenceImg_gpu(const PtrStepSzf& next,const PtrStepSzf& prev,PtrStepSzf diff)
{
//    static int num = 0;
//    differenceImg_gpu<<<1,1>>>();
//    std::cout<<"static function "<<num++<<std::endl;
//    diff.create(next.rows,next.cols,next.type());
    dim3 Block(32,8);
    dim3 Grid(iDivUp(next.step/next.elemSize(),Block.x),iDivUp(next.rows,Block.y));
    differenceImg_gpu1<<<Grid,Block>>>(next,prev,diff,next.step/next.elemSize());

    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
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
#if 1
        Mat gray_fpt_cpu(gray_fpt);
        cv::resize(gray_fpt_cpu, gray_fpt_cpu, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
        gray_fpt.upload(gray_fpt_cpu);
#else
        cuda::resize(gray_fpt, gray_fpt, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
#endif
//                    GpuMat test;
//                    gray_fpt.convertTo(test, DataType<uchar>::type, 1, 0);
//                    cv::Mat show(test);
//                    cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//                    cv::imshow("show",show);
//                    cv::waitKey(0);
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
#if 1
            cuda::subtract(next, prev, diff, noArray(), DataType<float>::type);
#else
            diff.create(next.rows,next.cols,next.type());
            differenceImg_gpu(next,prev,diff);
#endif
//            GpuMat test;
//            diff.convertTo(test, DataType<uchar>::type, 100, 0);
//            cv::Mat show(test);
//            cv::namedWindow("show",WINDOW_GUI_EXPANDED);
//            cv::imshow("show",show);
//            cv::waitKey(0);
        }
    }
}
//three question:
//1.keypoint allocate with unknow size: using GpuMat allocate
//ensureSizeIsEnough(SURF_CUDA::ROWS_COUNT, maxFeatures, CV_32FC1, keypoints);
//2.Two dimension vector:vector<GpuMat> send to kernel: using
//3.The sum of the keypoints num: using atomAdd()

void findScaleSpaceExtrema(std::vector<GpuMat>& gpyr, std::vector<GpuMat>& dogpyr, GpuMat& keypoints,GpuMat& descriptorsGpu,float contrastThreshold,int nOctaveLayers,int maxFeatures)
{

    ensureSizeIsEnough(SIFT_CUDA::ROWS_COUNT, maxFeatures, CV_32FC1, keypoints);
    keypoints.setTo(Scalar::all(0));

    int featureNum = 0;
    safeCall(cudaMemcpyToSymbol(d_PointCounter, &featureNum, sizeof(int)));

    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    float **h_pd = new float*[dogpyr.size()];
    for(int i = 0;i<dogpyr.size();i++)
        h_pd[i] = (float*)dogpyr[i].ptr();
    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*dogpyr.size()));


    float **h_gpyr = new float*[gpyr.size()];
    for(int i = 0;i<gpyr.size();i++)
        h_gpyr[i] = (float*)gpyr[i].ptr();
    safeCall(cudaMemcpyToSymbol(pgpyr, h_gpyr, sizeof(float *)*gpyr.size()));


    int temDataSize = 0;
    safeCall(cudaMemcpyToSymbol(temsize, &temDataSize, sizeof(int)));


    dim3 Block(32,8);
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers;i++){
            int index = o*(nOctaveLayers+2)+i+1;
            dim3 Grid(iDivUp(dogpyr[index].step1(),Block.x),iDivUp(dogpyr[index].rows,Block.y));
            findScaleSpaceExtrema_gpu<<<Grid,Block>>>((float*)keypoints.ptr(),keypoints.step1(),index,dogpyr[index].cols,dogpyr[index].step1(),dogpyr[index].rows,threshold,nOctaveLayers,maxFeatures);
            CV_CUDEV_SAFE_CALL( cudaGetLastError() );
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    }


    safeCall(cudaMemcpyFromSymbol(&featureNum, d_PointCounter, sizeof(int)));
    featureNum = (featureNum>maxFeatures)? maxFeatures:featureNum;
    //printf("cuda sift kepoints num : %d \n",featureNum);

    int* oIndex = new int[33];
    for(int i =0;i<nOctaves;i++){
        int index = i*(nOctaveLayers+2);
        oIndex[i*3] = dogpyr[index].cols;
        oIndex[i*3+1] = dogpyr[index].rows;
        oIndex[i*3+2] = dogpyr[index].step1();
    }
    safeCall(cudaMemcpyToSymbol(d_oIndex, oIndex, sizeof(int)*33));

    float* temData;
    safeCall(cudaMemcpyFromSymbol(&temDataSize, temsize, sizeof(int)));
    //4 is the 4 len buf
    int buffSize = temDataSize*4;
    safeCall(cudaMalloc(&temData,sizeof(float)*featureNum*buffSize));
    //std::cout<<"buffSize:"<<buffSize<<std::endl;

    int grid =iDivUp(featureNum,BLOCK_SIZE_ONE_DIM);
    //use the global memory
    //calcOrientationHist_gpu<<<grid,BLOCK_SIZE_ONE_DIM>>>(d_keypoints,temData,buffSize,num0,maxPoints,nOctaveLayers);
    calcOrientationHist_gpu<<<grid,BLOCK_SIZE_ONE_DIM>>>((float*)keypoints.ptr(),keypoints.step1(),temData,buffSize,featureNum,maxFeatures,nOctaveLayers);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    cudaFree(temData);

    //int num1 = 0;
    //featureNum is after calculate the orientation
    safeCall(cudaMemcpyFromSymbol(&featureNum, d_PointCounter, sizeof(int)));
    featureNum = (featureNum>maxFeatures)? maxFeatures:featureNum;
    //printf("cuda sift kepoints num after calOritation : %d \n",num1);

    keypoints.cols = featureNum;





    //Mat descriptors;
    //descriptors.create(featureNum,128,CV_32FC1);
    //safeCall(cudaMemcpy((float*)descriptors.data,d_descriptor,featureNum*128*sizeof(float),cudaMemcpyDeviceToHost));

//    descriptorsGpu.create(descriptors.rows,descriptors.cols,CV_32FC1);
//    descriptorsGpu.upload(descriptors);

    //cudaFree(d_descriptor);




//    Mat keypointsCPU(keypoints);
//    float* h_keypoints = (float*)keypointsCPU.ptr();
//    std::vector<cv::KeyPoint>keypointss;
//    keypointss.resize(num1);
//    for(int i = 0;i<keypointss.size();++i)
//    {
//        keypointss[i].pt.x =  h_keypoints[i];
//        keypointss[i].pt.y =  h_keypoints[i+keypointsCPU.step1()*1];
//        keypointss[i].octave =  h_keypoints[i+keypointsCPU.step1()*2];
//        keypointss[i].size =  h_keypoints[i+keypointsCPU.step1()*3];
//        keypointss[i].response =  h_keypoints[i+keypointsCPU.step1()*4];
//        keypointss[i].angle =  h_keypoints[i+keypointsCPU.step1()*5];
//    }
//    int firstOctave = -1;
//    if( firstOctave < 0 )
//        for( size_t i = 0; i < keypointss.size(); i++ )
//        {
//            KeyPoint& kpt = keypointss[i];
//            float scale = 1.f/(float)(1 << -firstOctave);
//            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
//            kpt.pt *= scale;
//            kpt.size *= scale;
//        }
//    Mat kepoint;
//    Mat dst(gpyr[6]),img;
//    dst.convertTo(img, DataType<uchar>::type, 1, 0);
//    drawKeypoints(img, keypointss,kepoint,cv::Scalar::all(-1),4);
//    cvNamedWindow("new cuda sift",CV_WINDOW_NORMAL);
//    imshow("new cuda sift", kepoint);
//    //等待任意按键按下
//    //waitKey(0);

//    Mat ss;
//    descriptors.convertTo(ss, DataType<uchar>::type, 1, 0);
//    cvNamedWindow("new descriptors",CV_WINDOW_NORMAL);
//    imshow("new descriptors", ss);


}
void calcDescriptors(GpuMat& keypoints,GpuMat& descriptorsGpu,int nOctaveLayers)
{
    //alloc for d_decriptor
    int featureNum = keypoints.cols;
    int despriptorSize = SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
    ensureSizeIsEnough(featureNum, despriptorSize, CV_32F, descriptorsGpu);
    //float* d_descriptor;

    //cudaMalloc(&d_descriptor,sizeof(float)*featureNum*despriptorSize);


    int grid =iDivUp(featureNum,Descript_BLOCK_SIZE);
    calcSIFTDescriptor_gpu<<<grid,Descript_BLOCK_SIZE>>>((float*)keypoints.ptr(),keypoints.step1(),(float*)descriptorsGpu.ptr(),featureNum,nOctaveLayers);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
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


        sift.maxFeatures = std::min(static_cast<int>(img.size().area() * sift.keypointsRatio), 65535);


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
        if( !this->sift.upsample )
            firstOctave = 0;
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
        //t = (double)getTickCount() - t;
        //printf("pyramid construction time: %g\n", t*1000./tf);
        buildDoGPyramid(gpyr, dogpyr,sift.nOctaveLayers);
        if( !useProvidedKeypoints )
        {
            //t = (double)getTickCount();
            findScaleSpaceExtrema(gpyr, dogpyr, keypoints,descriptors,sift.contrastThreshold,sift.nOctaveLayers,sift.maxFeatures);
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
        }
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
        calcDescriptors(keypoints,descriptors,sift.nOctaveLayers);
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
    SIFT_CUDA::SIFT_CUDA(bool _upsample, int _nOctaveLayers,
                         float _contrastThreshold, float _edgeThreshold,
                         float _sigma, float _keypointsRatio)
    {
        upsample = _upsample;
        nOctaveLayers = _nOctaveLayers;
        contrastThreshold = _contrastThreshold;
        edgeThreshold = _edgeThreshold;
        sigma = _sigma;
        keypointsRatio = _keypointsRatio;

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

            //sift.computeDescriptors(keypoints, descriptors);
        }


    }

    void SIFT_CUDA::downloadKeypoints(const GpuMat& keypointsGPU, std::vector<KeyPoint>& keypoints)
    {
        const int nFeatures = keypointsGPU.cols;

        if (nFeatures == 0)
            keypoints.clear();
        else
        {
            CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == ROWS_COUNT);

            Mat keypointsCPU(keypointsGPU);

            keypoints.resize(nFeatures);

            float* kp_x = keypointsCPU.ptr<float>(SIFT_CUDA::X_ROW);
            float* kp_y = keypointsCPU.ptr<float>(SIFT_CUDA::Y_ROW);
            int* kp_octave = keypointsCPU.ptr<int>(SIFT_CUDA::OCTAVEANDLAYER_ROW);
            float* kp_size = keypointsCPU.ptr<float>(SIFT_CUDA::SIZE_ROW);
            int* kp_response = keypointsCPU.ptr<int>(SIFT_CUDA::RESPONSE_ROW);
            float* kp_dir = keypointsCPU.ptr<float>(SIFT_CUDA::ANGLE_ROW);


            for (int i = 0; i < nFeatures; ++i)
            {
                KeyPoint& kp = keypoints[i];
                kp.pt.x = kp_x[i];
                kp.pt.y = kp_y[i];
                kp.octave = kp_octave[i];
                kp.size = kp_size[i];
                kp.response = kp_response[i];
                kp.angle = kp_dir[i];
            }

            if( this->upsample ){
                int firstOctave = -1;
                for( size_t i = 0; i < nFeatures; i++ )
                {
                    KeyPoint& kpt = keypoints[i];
                    float scale = 1.f/(float)(1 << -firstOctave);
                    kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                    kpt.pt *= scale;
                    kpt.size *= scale;
                }
            }
        }


    }





}

}
