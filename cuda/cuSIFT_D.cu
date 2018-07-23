//#include "cuImage.h"

//namespace cusift {
///******************************* Defs and macros *****************************/

//// default width of descriptor histogram array
//static const int SIFT_DESCR_WIDTH = 4;

//// default number of bins per histogram in descriptor array
//static const int SIFT_DESCR_HIST_BINS = 8;

//// assumed gaussian blur for input image
//static const float SIFT_INIT_SIGMA = 0.5f;

//// width of border in which to ignore keypoints
//static const int SIFT_IMG_BORDER = 5;

//// maximum steps of keypoint interpolation before failure
//static const int SIFT_MAX_INTERP_STEPS = 5;

//// default number of bins in histogram for orientation assignment
//static const int SIFT_ORI_HIST_BINS = 36;

//// determines gaussian sigma for orientation assignment
//static const float SIFT_ORI_SIG_FCTR = 1.5f;

//// determines the radius of the region used in orientation assignment
//static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

//// orientation magnitude relative to max that results in new feature
//static const float SIFT_ORI_PEAK_RATIO = 0.8f;

//// determines the size of a single descriptor orientation histogram
//static const float SIFT_DESCR_SCL_FCTR = 3.f;

//// threshold on magnitude of elements of descriptor vector
//static const float SIFT_DESCR_MAG_THR = 0.2f;

//// factor used to convert floating-point descriptor to unsigned char
//static const float SIFT_INT_DESCR_FCTR = 512.f;

//static const int SIFT_FIXPT_SCALE = 1;

//}

//namespace cusift {

//void createInitialImage_gpu(const Mat &src, CudaImage &base, float sigma, bool doubleImageSize){
//    int width = src.cols;
//    int height = src.rows;
//    if(!src.data){
//        printf("input none data !");
//        return;
//    }

//    Mat gray, gray_fpt;
//    if( src.channels() == 3 || src.channels() == 4 )
//    {
//        cvtColor(src, gray, COLOR_BGR2GRAY);
//        gray.convertTo(gray_fpt, DataType<float>::type, 1, 0);
//    }
//    else
//        src.convertTo(gray_fpt, DataType<float>::type, 1, 0);

//    //sigma different which is sqrt(1.6*1.6-0.5*0.5*4)
//    float sig_diff;

//    if( doubleImageSize )
//    {
//        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
//        resize(gray_fpt, gray_fpt, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
//        width = gray_fpt.cols;
//        height = gray_fpt.rows;
//        base.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)gray_fpt.data);
//        base.Download();
//        //cuGaussianBlur(base,sig_diff);

//    }
//    else
//    {
//        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
//        base.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)gray_fpt.data);
//        base.Download();
//        //cuGaussianBlur(base,sig_diff);
//        //GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
//    }

//}

//}

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <device_functions.hpp>
#include <opencv2/cudev.hpp>

namespace cv { namespace cuda { namespace device
{
    namespace sift
    {
        //void bindImgTex(PtrStepSzb img);

        //void compute_descriptors_gpu(PtrStepSz<float4> descriptors, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir, int nFeatures);
        //extern void differenceImg_gpu(PtrStepSzf next,PtrStepSzf prev,PtrStepSzf diff);
        //void differenceImg_gpu();
    }
}}}



namespace cv { namespace cuda { namespace device
{
namespace sift
{
using namespace cv::cuda;


__global__ void differenceImg_gpu1(PtrStepSzf next,PtrStepSzf prev,PtrStepSzf diff,int pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y<next.rows)
    {
        diff(y,x) = next(y,x)-prev(y,x);

    }
    if(y*pitch+x<5)
        printf("%f\n",next(y,x));

}

__device__ unsigned int d_PointCounter[1];
//choose 60 suport 16384 pixel size image (log2(16384) - 2)*5
__device__ float *pd[60];
//choose 72 suport 16384 pixel size image (log2(16384) - 2)*6
__device__ float *pgpyr[72];
__device__ int temsize;
//36 suppose the max Octave is 12
__constant__ int d_oIndex[36];

static const int BLOCK_SIZE_ONE_DIM = 32;

__global__ void test_gpu(int pitch,int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y<height)
    {
        pgpyr[0][y*pitch+x] += 100;
    }
    if(y*pitch+x<5)
        printf("%f\n",pd[0][y*pitch+x]);
}

__global__ void findScaleSpaceExtrema_gpu(float *d_point,int p_pitch,int s, int width ,int pitch ,int height,const int threshold,const int nOctaveLayers,const int maxNum){

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    //avoid extract the unstable border points
    if(y >= height - SIFT_IMG_BORDER || x >= width - SIFT_IMG_BORDER || x<SIFT_IMG_BORDER || y<SIFT_IMG_BORDER)
        return;

    float *currptr = pd[s]  +y*pitch+x;
    float *prevptr = pd[s-1]+y*pitch+x;
    float *nextptr = pd[s+1]+y*pitch+x;

    int o = s/(nOctaveLayers+2);
    float val = *currptr;
    int step = pitch;
    int c = 0;
    if( std::abs(val) > threshold &&
       ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
         val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
         val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
         val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
         val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
         val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
         val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
         val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
         val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
        (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
         val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
         val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
         val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
         val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
         val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
         val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
         val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
         val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
    {
        /*adjustLocalExtrema*/
        const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
        const float deriv_scale = img_scale*0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale = img_scale*0.25f;
        float Vs=0, Vx=0, Vy=0, contr=0;
        float dx,dy,ds,dxx,dyy,dxy;
        int j = 0,layer;
        //get the x,y,s,Vs,Vx,Vy or return
        for( ; j < SIFT_MAX_INTERP_STEPS; j++ )
        {
            currptr = pd[s]  +y*pitch+x;
            prevptr = pd[s-1]+y*pitch+x;
            nextptr = pd[s+1]+y*pitch+x;

            //the first derivative of x,y and scale
            dx = (currptr[1] - currptr[-1])*deriv_scale;
            dy = (currptr[pitch] - currptr[-pitch])*deriv_scale;;
            ds = (nextptr[0] - prevptr[0])*deriv_scale;
            float v2 = currptr[0]*2;

            //the second derivative of x,y,scale
            dxx = (currptr[1] + currptr[-1] - v2)*second_deriv_scale;
            dyy = (currptr[pitch] + currptr[-pitch] - v2)*second_deriv_scale;
            float dss = (nextptr[0] + prevptr[0] - v2)*second_deriv_scale;
            dxy = (currptr[pitch+1] - currptr[1-pitch] -
                         currptr[-1+pitch] + currptr[-pitch-1])*cross_deriv_scale;
            float dxs = (nextptr[1] - nextptr[-1] -
                         prevptr[1] + prevptr[-1])*cross_deriv_scale;
            float dys = (nextptr[pitch] - nextptr[-pitch] -
                         prevptr[pitch] + prevptr[-pitch])*cross_deriv_scale;

            //Algebraic cousin
            float idxx = dyy*dss - dys*dys;
            float idxy = dys*dxs - dxy*dss;
            float idxs = dxy*dys - dyy*dxs;
            //idet is the det,the matrix's determinant countdown
            float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
            float idyy = dxx*dss - dxs*dxs;
            float idys = dxy*dxs - dxx*dys;
            float idss = dxx*dyy - dxy*dxy;
            ////////////////////////
            ///  A(dxx, dxy, dxs,
            ///    dxy, dyy, dys,
            ///    dxs, dys, dss);
            ///
            ///  A*(idxx, idxy, idxs,
            ///     idxy, idyy, idys,
            ///     idxs, idys, idss);
            ///
            ///  B(dx,dy,dz)
            /////////////////////////
            //dX = (A^-1)*B
            float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
            float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
            float pds = idet*(idxs*dx + idys*dy + idss*ds);

            Vx = -pdx;
            Vy = -pdy;
            Vs = -pds;

            //because of the judgment is before the updated value,so
            //this iteration final get the x,y,s(intger) and the Vx,Vy,Vz(<0.5).
            //The accurate extrema location is x+Vx,y+Vy.

            if( std::abs(Vs) < 0.5f && std::abs(Vx) < 0.5f && std::abs(Vy) < 0.5f )
                break;

            //get nearest intger
            x += int(Vx > 0 ? ( Vx + 0.5 ) : (Vx - 0.5));
            y += int(Vy > 0 ? ( Vy + 0.5 ) : (Vy - 0.5));
            s += int(Vs > 0 ? ( Vs + 0.5 ) : (Vs - 0.5));

            layer = s - o*(nOctaveLayers+2);

            if( layer < 1 || layer > nOctaveLayers ||
                y < SIFT_IMG_BORDER || y >= height - SIFT_IMG_BORDER  ||
                x < SIFT_IMG_BORDER || x >= width - SIFT_IMG_BORDER )
                return;

        }//for
        if( j >= SIFT_MAX_INTERP_STEPS )
            return;

        //After the iterative,get the x,y,s,(Vx,Vy,Vs)(<0.5).

        {
            //remove the small energy points which essily influenced by image noise
            float t = dx*Vx + dy*Vy + ds*Vs;
            contr = currptr[0]*img_scale + t * 0.5f;
            if( std::abs( contr ) * nOctaveLayers < 0.04 )
                return;

            // principal curvatures are computed using the trace and det of Hessian
            float tr = dxx + dyy;
            float det = dxx*dyy-dxy*dxy;

            if( det <= 0 || tr*tr*10 >= (10 + 1)*(10 + 1)*det )
                return;
        }

        layer = s - o*(nOctaveLayers+2);

        //addpoint;
        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
        idx = (idx>maxNum ? maxNum-1 : idx);
//        d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
//        d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
//        d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
//        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
//        d_point[idx*KEYPOINTS_SIZE+3] = size;
//        d_point[idx*KEYPOINTS_SIZE+4] = std::abs(contr);
//        d_point[idx*KEYPOINTS_SIZE+6] = s;
//        d_point[idx*KEYPOINTS_SIZE+7] = x;
//        d_point[idx*KEYPOINTS_SIZE+8] = y;

        d_point[idx] = (x + Vx)*(1 << o);
        d_point[idx+p_pitch*1] = (y + Vy)*(1 << o);
        float oct_lay1 =o + (layer<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
        int oct_lay = oct_lay1;
        d_point[idx+p_pitch*2] = oct_lay1;
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
        d_point[idx+p_pitch*3] = size;
        d_point[idx+p_pitch*4] = std::abs(contr);
//        int _octave,_layer;
//        _octave = oct_lay & 255;
//        layer = (oct_lay >> 8) & 255;
//        _octave = _octave < 128 ? _octave : (-128 | _octave);
//        s = _octave*(nOctaveLayers+2)+layer;
//        x = round(d_point[idx]/(1<<_octave));
//        y = round(d_point[idx+p_pitch*1]/(1<<_octave));
//        d_point[idx+p_pitch*6] = s;
//        d_point[idx+p_pitch*7] = x;
//        d_point[idx+p_pitch*8] = y;

        //temsize+=size*0.5f/(1 << o)*SIFT_ORI_RADIUS+0.5;
//        if(x<2000 && y<2000)
//            printf("%d,%d,%d\n",x,y,s);
        //printf("%f \n",pd[0][100*2304+100]);

        float scl_octv = size*0.5f/(1 << o);
        //'+0.5' for rounding because scl_octv>0
        int radius = SIFT_ORI_RADIUS * scl_octv+0.5;
        //the procress of all point range, a square space.
        int len = (radius*2+1)*(radius*2+1);
        //int temBuffSize = len*4+2*SIFT_ORI_HIST_BINS+2;
        atomicMax(&temsize,len);
    }
}

__device__ void unpackOctave(float& fx,float& fy,float& oct_lay1,int& x,int& y,int& octave,int& layer)
{
    int oct_lay = oct_lay1;
    octave = oct_lay & 255;
    layer = (oct_lay >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    x = round(fx/(1<<octave));
    y = round(fy/(1<<octave));
}

__global__ void calcOrientationHist_gpu(float *d_point,int p_pitch,float* temdata,const int buffSize,const int pointsNum,const int maxNum,const int nOctaveLayers)
{
    //int x = blockIdx.x*blockDim.x+threadIdx.x;
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;

    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex];
    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex+p_pitch*1];
    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex+p_pitch*2];
    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex+p_pitch*3];
    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex+p_pitch*4];
    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex+p_pitch*5];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex+p_pitch*6];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex+p_pitch*7];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex+p_pitch*8];

    __syncthreads();
    float size =s_point[threadIdx.x*KEYPOINTS_SIZE+3];

    int x,y,o,layer;
    unpackOctave(s_point[threadIdx.x*KEYPOINTS_SIZE],
            s_point[threadIdx.x*KEYPOINTS_SIZE+1],
            s_point[threadIdx.x*KEYPOINTS_SIZE+2],
            x,y,o,layer);

//    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
//    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
//    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
//    int o = s/(nOctaveLayers+2);
//    int layer = s - o*(nOctaveLayers+2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];


    float* currptr;
    //currptr is the current dog image where the current extrema point in.
    //x,y,s is the current location in dog images.
    //Note: s is the absolutely scale location and the 'laryer' is the /
    //relatively location in the octave which range is 1~3.

    //The orientation is compute in gausspyrmid,so the currptr renew:
    currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;
    //simga*2^s/S,the simga the simga relative to the octave.
    float scl_octv = size*0.5f/(1 << o);
    float omax;
    float sigma_ori = SIFT_ORI_SIG_FCTR * scl_octv;
    //'+0.5' for rounding because scl_octv>0
    int radius = SIFT_ORI_RADIUS * scl_octv+0.5,n = SIFT_ORI_HIST_BINS;
    float* hists = new float[2*n+4];

    //the procress of all point range, a square space.
    int len = (radius*2+1)*(radius*2+1);
    //garuss smooth's coefficient
    float expf_scale = -1.f/(2.f * sigma_ori * sigma_ori);
    //n = 36
    float *buf = temdata+pointIndex*buffSize;
    //float *buf = (float *)malloc((len*4 + n+4 + n)*sizeof(float));
    //the buf is a memory storage the temporary data.
    //The frist len is the Mag('fu zhi')and X,second len is the Y,third len is the Ori,
    //the forth is gauss weight(len+2)
    //the temphist is(n + 2).
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    //gradient direction histogarm
    float* temphist = hists + 2;
    float* hist = temphist + 2+n;

    for( int i = 0; i < n; i++ )
        temphist[i] = 0.f;

//    if(radius > 16)
//        printf("radius: %d, point index : %d\n",radius,pointIndex);

    for( int i = -radius, k = 0; i <= radius; i++ )
    {
        int yi = y + i;
        // '=' avoid out of memory for i-1,j-1 following
        if( yi <= 0 || yi >= height - 1 )
            continue;
        for( int j = -radius; j <= radius; j++ )
        {
            int xi = x + j;
            if( xi <= 0 || xi >= width - 1 )
                continue;

            float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
            //the positive direction is from bottom to top contrary to the image /
            //from top to bottom.So dy = y-1 - (y+1).
            float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);

            //X[k] = dx;
            //Y[k] = dy;
            //Wight not multiply 1/pi,because the compute of oritentation
            //only need the relative wight.
            float wk,ok,mk;
//            W[k] = __expf((i*i + j*j)*expf_scale);
//            Ori[k] = atan2f(dy,dx);
//            Mag[k] = sqrtf(dy*dy+dx*dx);
            wk = __expf((i*i + j*j)*expf_scale);
            ok = atan2f(dy,dx);
            mk = sqrtf(dy*dy+dx*dx);
            //cvRound((ori/pi+180)/360*36)
            float tembin = __fdividef(__fdividef(ok*180,CV_PI),360/n);
            int bin = tembin > 0 ? tembin + 0.5:tembin - 0.5;
            if( bin >= n )
                bin -= n;
            if( bin < 0 )
                bin += n;
            temphist[bin] += wk*mk;
            k++;
        }
    }

    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    for(int i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    omax = hist[0];
    for( int i = 1; i < n; i++ )
        omax = fmaxf(omax, hist[i]);
    //printf("omax : %f \n",omax);

    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);

    for( int j = 0; j < n; j++ )
    {
        int l = j > 0 ? j - 1 : n - 1;
        int r2 = j < n-1 ? j + 1 : 0;

        if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
        {
            float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
            bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;

            if(hist[j] == omax)
                d_point[pointIndex+p_pitch*5] = 360.f - (float)((360.f/n) * bin);
            else{
                //addpoint;
                unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                idx = (idx>maxNum ? maxNum-1 : idx);
                d_point[idx]   = s_point[threadIdx.x*KEYPOINTS_SIZE];
                d_point[idx+p_pitch*1] = s_point[threadIdx.x*KEYPOINTS_SIZE+1];
                d_point[idx+p_pitch*2] = s_point[threadIdx.x*KEYPOINTS_SIZE+2];
                d_point[idx+p_pitch*3] = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
                d_point[idx+p_pitch*4] = s_point[threadIdx.x*KEYPOINTS_SIZE+4];
                d_point[idx+p_pitch*5] = 360.f - (float)((360.f/n) * bin);
//                d_point[idx+p_pitch*6] = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
//                d_point[idx+p_pitch*7] = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
//                d_point[idx+p_pitch*8] = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
                //printf("%f ",360.f - (float)((360.f/n) * bin));
            }
        }
    }

    delete []hists;
}

__global__ void calcSIFTDescriptor_gpu_bk(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;
//    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

//    __syncthreads();
//    float size = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
//    float ori = s_point[threadIdx.x*KEYPOINTS_SIZE+5];
//    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
//    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
//    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    __shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
    float* dst = dst1+threadIdx.x*d*d*n;
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
    //float *buf = new float[len*6 + histlen];
    const int histlen = (SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_HIST_BINS+2);
    float hist[histlen];
    //__shared__ float hist[histlen*BLOCK_SIZE_ONE_DIM];


    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");

                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;

                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
                hist[idx] += v_rco000;
                hist[idx+1] += v_rco001;
                hist[idx+(n+2)] += v_rco010;
                hist[idx+(n+3)] += v_rco011;
                hist[idx+(d+2)*(n+2)] += v_rco100;
                hist[idx+(d+2)*(n+2)+1] += v_rco101;
                hist[idx+(d+3)*(n+2)] += v_rco110;
                hist[idx+(d+3)*(n+2)+1] += v_rco111;

                k++;
            }
        }
//    if(x == 1936 && y ==744 ){
//        for(int i =0;i<360;i++)
//            printf(" %f ",hist[i]);
//        printf("k: %d",k);
//    }


    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
               dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    __syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}

__device__ int judgeValue(int idx,int height_360 ,int col_360 ,int d,int n)
{
    int row_Col = idx/height_360;
    int z = idx%height_360;
    int y = row_Col/col_360;
    int x = row_Col%col_360;
//    if(idx == 70)
//    {
//        printf(" %d %d %d ",x,y,z);
//    }
    if( x>0 && y>0 && x<d+1 && y<d+1 && z < n+1)
    {
        if(z == n)
        {
            z = 0;
        }
        return ((y-1)*(d)+x-1)*(n) + z;
//        if(((y-1)*(d)+x-1)*(n) + z == 6)
//            printf(" %f: ",value);
    }
    return -1;
}

__global__ void calcSIFTDescriptor_gpu_bk__(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;
//    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

//    __syncthreads();
//    float size = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
//    float ori = s_point[threadIdx.x*KEYPOINTS_SIZE+5];
//    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
//    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
//    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    __shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
    float* dst = dst1+threadIdx.x*d*d*n;
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
    //float *buf = new float[len*6 + histlen];
    const int histlen = (SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_HIST_BINS+2);
    float hist[histlen];
    //__shared__ float hist[histlen*BLOCK_SIZE_ONE_DIM];


    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

                hist[idx] += v_rco000;
                hist[idx+1] += v_rco001;
                hist[idx+(n+2)] += v_rco010;
                hist[idx+(n+3)] += v_rco011;
                hist[idx+(d+2)*(n+2)] += v_rco100;
                hist[idx+(d+2)*(n+2)+1] += v_rco101;
                hist[idx+(d+3)*(n+2)] += v_rco110;
                hist[idx+(d+3)*(n+2)+1] += v_rco111;
//                int idx1 = judgeValue(idx,n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco000;
//                idx1 = judgeValue(idx+1,n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco010;
//                idx1 = judgeValue(idx+(n+2)+(n+3),n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco011;
//                idx1 = judgeValue(idx+(n+3),n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco100;
//                idx1 = judgeValue(idx+(d+2)*(n+2),n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco101;
//                idx1 = judgeValue(idx+(d+2)*(n+2)+1,n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco110;
//                idx1 = judgeValue(idx+(d+3)*(n+2),n+2,d+2,d,n);
//                if(idx1>0)
//                    hist[idx1] += v_rco111;
                k++;
            }
        }
//    if(x == 1936 && y ==744 ){
//        for(int i =0;i<360;i++)
//            printf(" %f ",hist[i]);
//        printf("k: %d",k);
//    }


    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
               dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    __syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}

__device__ void fillValue_128(float* hist,int idx,float value,int height_360 ,int col_360 ,int d,int n)
{
    int row_Col = idx/height_360;
    int z = idx%height_360;
    int y = row_Col/col_360;
    int x = row_Col%col_360;
//    if(idx == 70)
//    {
//        printf(" %d %d %d ",x,y,z);
//    }
    if( x>0 && y>0 && x<d+1 && y<d+1 && z < n+1)
    {
        if(z == n)
        {
            z = 0;
        }
        hist[((y-1)*(d)+x-1)*(n) + z] += value;
//        if(((y-1)*(d)+x-1)*(n) + z == 6)
//            printf(" %f: ",value);
    }

//    if( x==1 && y==1 && z == n+1)
//    {
//        if(z == n)
//        {
//            z = 0;
//        }
//        hist[((y-1)*(d)+x-1)*(n) + z] += value;
////        if(((y-1)*(d)+x-1)*(n) + z == 6)
////            printf(" %f: ",value);
//    }

}

__global__ void calcSIFTDescriptor_gpu__test(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;
//    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
//    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

//    __syncthreads();
//    float size = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
//    float ori = s_point[threadIdx.x*KEYPOINTS_SIZE+5];
//    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
//    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
//    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
//    __shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
//    float* dst = dst1+threadIdx.x*d*d*n;
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
    //float *buf = new float[len*6 + histlen];
    const int histlen = (SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS);
    __shared__ float dst1[histlen*BLOCK_SIZE_ONE_DIM];
    float* hist = dst1+threadIdx.x*d*d*n;
    //__shared__ float hist[histlen*BLOCK_SIZE_ONE_DIM];


    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d; i++ )
    {
        for( j = 0; j < d; j++ )
            for( k = 0; k < n; k++ )
                hist[(i*(d) + j)*(n) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d-1 && cbin > -1 && cbin < d-1 &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;


                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d",idx);
//                }
//                idx = idx>80 ? idx - 50:idx;
                fillValue_128(hist,idx,v_rco000,n+2,d+2,d,n);
                fillValue_128(hist,idx+1,v_rco001,n+2,d+2,d,n);
                fillValue_128(hist,idx+(n+2),v_rco010,n+2,d+2,d,n);
                fillValue_128(hist,idx+(n+3),v_rco011,n+2,d+2,d,n);
                fillValue_128(hist,idx+(d+2)*(n+2),v_rco100,n+2,d+2,d,n);
                fillValue_128(hist,idx+(d+2)*(n+2)+1,v_rco101,n+2,d+2,d,n);
                fillValue_128(hist,idx+(d+3)*(n+2),v_rco110,n+2,d+2,d,n);
                fillValue_128(hist,idx+(d+3)*(n+2)+1,v_rco111,n+2,d+2,d,n);

                k++;
            }
        }
//    if(x == 1936 && y ==744 ){
//        for(int i =0;i<360;i++)
//            printf(" %f ",hist[i]);
//        printf("k: %d",k);
//    }

    float* dst = hist;
//    float nrm2 = 1;
//    len = 128;
    // finalize histogram, since the orientation histograms are circular
//    for( i = 0; i < d; i++ )
//        for( j = 0; j < d; j++ )
//        {
//            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
//            hist[idx] += hist[idx+n];
//            hist[idx+1] += hist[idx+n+1];
//            for( k = 0; k < n; k++ )
//               dst[(i*d + j)*n + k] = hist[idx+k];
//        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array

    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    __syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}


__global__ void calcSIFTDescriptor_gpu_debug(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    __shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
    float* dst = dst1+threadIdx.x*d*d*n;
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
    //float *buf = new float[len*6 + histlen];
    const int histlen = (SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_WIDTH+2)*(SIFT_DESCR_HIST_BINS+2);
    float hist[histlen];


    //__shared__ float hist[histlen*BLOCK_SIZE_ONE_DIM];


    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,v_rco001,v_rco010,v_rco011,v_rco100,v_rco101);
//                }

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
                hist[idx] += v_rco000;
                hist[idx+1] += v_rco001;
                hist[idx+(n+2)] += v_rco010;
                hist[idx+(n+3)] += v_rco011;
                hist[idx+(d+2)*(n+2)] += v_rco100;
                hist[idx+(d+2)*(n+2)+1] += v_rco101;
                hist[idx+(d+3)*(n+2)] += v_rco110;
                hist[idx+(d+3)*(n+2)+1] += v_rco111;

                k++;
            }
        }
//    if(x == 1936 && y ==744 ){
//        for(int i =0;i<360;i++)
//            printf(" %f ",hist[i]);
//        printf("k: %d",k);
//    }


    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            //hist[idx+n+1] always = 0.,n+2 is useless, n+1 is engough and necessary
            //because z = 7.5 it will contribute or interpolate to (x,y,7.0) and (x,y,8)
            //z = 8 i.e. z = 0; so n+1 is necessary and hist[idx+1] += hist[idx+n+1]; is
            //right.
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
//            if(hist[idx+n + 1] != 0.)
//                printf("asa");
            for( k = 0; k < n; k++ )
               dst[(i*d + j)*n + k] = hist[idx+k];
        }


    const int histlen1 = (SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS);
    float hist1[histlen1];

    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d; i++ )
    {
        for( j = 0; j < d; j++ )
            for( k = 0; k < n; k++ )
                hist1[(i*(d) + j)*(n) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,v_rco001,v_rco010,v_rco011,v_rco100,v_rco101);
//                }

                fillValue_128(hist1,idx,v_rco000,n+2,d+2,d,n);
                fillValue_128(hist1,idx+1,v_rco001,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+2),v_rco010,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+3),v_rco011,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2),v_rco100,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2)+1,v_rco101,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2),v_rco110,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2)+1,v_rco111,n+2,d+2,d,n);
//                hist[idx] += v_rco000;
//                hist[idx+1] += v_rco001;
//                hist[idx+(n+2)] += v_rco010;
//                hist[idx+(n+3)] += v_rco011;
//                hist[idx+(d+2)*(n+2)] += v_rco100;
//                hist[idx+(d+2)*(n+2)+1] += v_rco101;
//                hist[idx+(d+3)*(n+2)] += v_rco110;
//                hist[idx+(d+3)*(n+2)+1] += v_rco111;
                k++;
            }
        }

    float dd = 0.;
    for( k = 0; k < 128; k++ ){
        dd += std::abs(hist1[k] - dst[k]);
    }
    //printf(" %f ",dd);

    //if(x == 440*2 && y ==322*2 ){
//        for( k = 0; k < 128; k++ ){
//            if(std::abs(hist1[k] - dst[k]) > 0.0001)
//                printf(" %d \n",k);
//        }
    //}



    if(x == 440*2 && y ==322*2 ){
        for( k = 0; k < 128; k++ ){
            printf(" %f ",std::abs(hist1[k]));
        }
        printf("\n[0][0] = %f \n",std::abs(hist1[0]));
    }

    if(x == 440*2 && y ==322*2 ){
        printf(" \n ");
        for( k = 0; k < 360; k++ ){
            printf(" %f ",std::abs(hist[k]));
        }
        printf("\n[0][0] = %f \n",std::abs(hist[70]));
    }

    if(x == 440*2 && y ==322*2 ){
        printf(" \n ");
        for( k = 0; k < 128; k++ ){
            printf(" %f ",std::abs(dst[k]));
        }
        printf("\n[0][0] = %f \n",std::abs(dst[0]));
    }
    //dst = hist1;
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    __syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}

__global__ void calcSIFTDescriptor_gpu(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    //__shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;

    const int histlen1 = (SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS);
    __shared__ float hist[(SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS)*BLOCK_SIZE_ONE_DIM];
    float* hist1 = hist+threadIdx.x*d*d*n;
    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d; i++ )
    {
        for( j = 0; j < d; j++ )
            for( k = 0; k < n; k++ )
                hist1[(i*(d) + j)*(n) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,v_rco001,v_rco010,v_rco011,v_rco100,v_rco101);
//                }
                //idx = 50;
                fillValue_128(hist1,idx,v_rco000,n+2,d+2,d,n);
                fillValue_128(hist1,idx+1,v_rco001,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+2),v_rco010,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+3),v_rco011,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2),v_rco100,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2)+1,v_rco101,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2),v_rco110,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2)+1,v_rco111,n+2,d+2,d,n);
//                hist1[int((idx)/360*128)] += v_rco000;
//                hist1[int((idx+1)/360*128)] += v_rco001;
//                hist1[int((idx+(n+2))/360*128)] += v_rco010;
//                hist1[int((idx+(n+3))/360*128)] += v_rco011;
//                hist1[int((idx+(d+2)*(n+2))/360*128)] += v_rco100;
//                hist1[int((idx+(d+2)*(n+2)+1)/360*128)] += v_rco101;
//                hist1[int((idx+(d+3)*(n+2))/360*128)] += v_rco110;
//                hist1[int((idx+(d+3)*(n+2)+1)/360*128)] += v_rco111;
//                atomicAdd(hist1+int((idx+(d+3)*(n+2)+1)/360*128), v_rco111);
//                atomicAdd(hist1+int((idx+(d+3)*(n+2))/360*128), v_rco110);
//                atomicAdd(hist1+int((idx+(d+2)*(n+2)+1)/360*128), v_rco100);
//                atomicAdd(hist1+int((idx+(n+3))/360*128), v_rco011);
//                atomicAdd(hist1+int((idx+(n+2))/360*128), v_rco010);
//                atomicAdd(hist1+int((idx+1)/360*128), v_rco001);
//                atomicAdd(hist1+int((idx)/360*128), v_rco000);
//                atomicAdd(hist1+int((idx+(d+3)*(n+2)+1)/360*128), v_rco111);
                k++;
            }
        }

    float *dst;
    dst = hist1;
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    //__syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}
__global__ void calcSIFTDescriptor_gpu_shared_(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    //__shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;

    const int histlen1 = (SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS);
    __shared__ float hist[(SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS)*BLOCK_SIZE_ONE_DIM];
    float* hist1 = hist+threadIdx.x*d*d*n;
    //float4 layer0[32];

    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d; i++ )
    {
        for( j = 0; j < d; j++ )
            for( k = 0; k < n; k++ )
                hist1[(i*(d) + j)*(n) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,v_rco001,v_rco010,v_rco011,v_rco100,v_rco101);
//                }
                //idx = 50;
                fillValue_128(hist1,idx,v_rco000,n+2,d+2,d,n);
                fillValue_128(hist1,idx+1,v_rco001,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+2),v_rco010,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+3),v_rco011,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2),v_rco100,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2)+1,v_rco101,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2),v_rco110,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2)+1,v_rco111,n+2,d+2,d,n);

//                int idx1 = judgeValue(idx,n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco000);
//                idx1 = judgeValue(idx+1,n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco001);
//                idx1 = judgeValue(idx+(n+2)+(n+3),n+2,d+2,d,n);
//                if(idx1>0)
//                     atomicAdd(hist1+idx1, v_rco010);
//                idx1 = judgeValue(idx+(n+3),n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco100);
//                idx1 = judgeValue(idx+(d+2)*(n+2),n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco101);
//                idx1 = judgeValue(idx+(d+2)*(n+2)+1,n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco110);
//                idx1 = judgeValue(idx+(d+3)*(n+2),n+2,d+2,d,n);
//                if(idx1>0)
//                    atomicAdd(hist1+idx1, v_rco111);




//                hist1[int((idx)/360*128)] += v_rco000;
//                hist1[int((idx+1)/360*128)] += v_rco001;
//                hist1[int((idx+(n+2))/360*128)] += v_rco010;
//                hist1[int((idx+(n+3))/360*128)] += v_rco011;
//                hist1[int((idx+(d+2)*(n+2))/360*128)] += v_rco100;
//                hist1[int((idx+(d+2)*(n+2)+1)/360*128)] += v_rco101;
//                hist1[int((idx+(d+3)*(n+2))/360*128)] += v_rco110;
//                hist1[int((idx+(d+3)*(n+2)+1)/360*128)] += v_rco111;
//                atomicAdd(hist1+int((idx+(d+3)*(n+2)+1)/360*128), v_rco111);
//                atomicAdd(hist1+int((idx+(d+3)*(n+2))/360*128), v_rco110);
//                atomicAdd(hist1+int((idx+(d+2)*(n+2)+1)/360*128), v_rco100);
//                atomicAdd(hist1+int((idx+(n+3))/360*128), v_rco011);
//                atomicAdd(hist1+int((idx+(n+2))/360*128), v_rco010);
//                atomicAdd(hist1+int((idx+1)/360*128), v_rco001);
//                atomicAdd(hist1+int((idx)/360*128), v_rco000);
//                atomicAdd(hist1+int((idx+(d+3)*(n+2)+1)/360*128), v_rco111);
                k++;
            }
        }

    float *dst;
    dst = hist1;
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    //__syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}

__global__ void calcSIFTDescriptor_gpu__local__(float *d_point,int p_pitch,float* d_decriptor,int pointsNum,int  nOctaveLayers)
{
    //float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index

    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;

    float size =d_point[pointIndex+p_pitch*3];
    float ori = d_point[pointIndex+p_pitch*5];


    int x,y,o,layer;
    unpackOctave(d_point[pointIndex],
            d_point[pointIndex+p_pitch*1],
            d_point[pointIndex+p_pitch*2],
            x,y,o,layer);

    float scl_octv = size/((1 << o)*2);

    int width = d_oIndex[o*3];
    int height = d_oIndex[o*3+1];
    int pitch = d_oIndex[o*3+2];

    float *currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;

    //printf(" %d,%d,%f,%f,%f ",x,y,*currptr,ori,scl_octv);
    //printf(" %d,%d,%f ",x,y,*(pgpyr[o*(nOctaveLayers+3) + layer]+1));

    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));

    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = fminf(radius, (int) sqrtf(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    //float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
//    __shared__ float dst1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS*BLOCK_SIZE_ONE_DIM];

//    for(int i = 0;i < 128 ; i++)
//    {
//        float* dst = dst1+threadIdx.x*d*d*n;
//        dst[i] = 0;
//    }
//    __syncthreads();
//    const int histlen1 = (SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS);
//    __shared__ float hist[(SIFT_DESCR_WIDTH)*(SIFT_DESCR_WIDTH)*(SIFT_DESCR_HIST_BINS)*BLOCK_SIZE_ONE_DIM];
//    float* hist1 = hist+threadIdx.x*d*d*n;

    float hist1[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];


    //init *hist = {0},because following code will use '+='
    for( i = 0; i < d; i++ )
    {
        for( j = 0; j < d; j++ )
            for( k = 0; k < n; k++ )
                hist1[(i*(d) + j)*(n) + k] = 0.;
    }

    //traverse the boundary rectangle
    //calculate two improtant data
    //1.all dx,dy,w,ori,mag in image coordinary
    //2.all x,y in bins coordinary(a related coordinary)
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.

            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = y + i, c = x + j;

            //d = 4
            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {

                float dx = (float)(currptr[i*pitch+j+1] - currptr[i*pitch+j-1]);
                //the positive direction is from bottom to top contrary to the image /
                //from top to bottom.So dy = y-1 - (y+1).
                float dy = (float)(currptr[(i-1)*pitch+j] - currptr[(i+1)*pitch+j]);
                // float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                // float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                //X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                float wk,ok,mk;
                wk = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                ok = atan2(dy,dx);
                //ok = cv::cudev::atan2();
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;



//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

//                if(o0 > n)
//                    printf("Aa");


                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;



                // histogram update using tri-linear interpolation
                float v_r1 = mag*rbin, v_r0 = mag - v_r1;
                float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }

//                if(x == 440*2 && y ==322*2 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,v_rco001,v_rco010,v_rco011,v_rco100,v_rco101);
//                }
                //idx = 50;
                fillValue_128(hist1,idx,v_rco000,n+2,d+2,d,n);
                fillValue_128(hist1,idx+1,v_rco001,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+2),v_rco010,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(n+3),v_rco011,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2),v_rco100,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+2)*(n+2)+1,v_rco101,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2),v_rco110,n+2,d+2,d,n);
                fillValue_128(hist1,idx+(d+3)*(n+2)+1,v_rco111,n+2,d+2,d,n);
//                if(idx+(d+1)*(n)+1>127)
//                    continue;
//                hist1[idx] += v_rco000;
//                hist1[idx+1] += v_rco001;
//                hist1[idx+(n)] += v_rco010;
//                hist1[idx+(n+1)] += v_rco011;
//                hist1[idx+(d)*(n)] += v_rco100;
//                hist1[idx+(d)*(n)+1] += v_rco101;
//                hist1[idx+(d+1)*(n)] += v_rco110;
//                hist1[idx+(d+1)*(n)+1] += v_rco111;
//                k++;
            }
        }

    float *dst;
    dst = hist1;
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];

    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;

    i = 0, nrm2 = 0;

    for( ; i < len; i++ )
    {
        float val = fminf(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    //__syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/fmaxf(sqrtf(nrm2), FLT_EPSILON);
    k = 0;
    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[pointIndex*len + k] = (uchar)(dst[k]*nrm2);
//        if(x == 21 && y ==257 ){
//            printf("k: %d,%f \n",k,d_decriptor[pointIndex*len + k]);
//        }
    }
}




}
}

}
}

