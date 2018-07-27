#include "cusitf_function_H.h"
using namespace cv;

#define MESSAGE 1

texture<float, 1, cudaReadModeElementType> texRef;





/***********
//This is an adjustable option which control the gaussKernel size. \
//when the kernel size less than 32*2+1 or kernel radius less than 32,the ROW_HALO_STEP set 1 \
//and the COLUMNS_HALO_STEPS set 2 will has a good performance.But when the kernel size is more \
//than 32 less than 64, the ROW_HALO_STEP should set 2 and the COLUMNS_HALO_STEPS should set 4
//The ROW_HALO_STEP will set 1 and the COLUMNS_HALO_STEPS will set 2 by default
***********/

///////////////////////
/// \brief GaussianBlurKernelRow
/// \param d_data
/// \param out
/// \param w
/// \param h
/// \param ksize
/// \param pitch
/// Only support the kernel size less than 32*2+1(ROW_HALO_STEP*ROW_BLOCK_DIM_X(32) is the radius)
/// Reference the cuda-sample 'convolutionSeparable'.
/// The boundary is set 0 which is different from OpenCV.The reason I simplify the boundary is \
/// that the description of the sift need not the boundary of the image which will be filter out.
/// If adjust the ROW_HALO_STEP 2,that is ok.
//////////////////////

#define   ROW_BLOCK_DIM_X 32
#define   ROW_BLOCK_DIM_Y 8
#define  ROW_UNROLL_STEPS 4
#define     ROW_HALO_STEP 1

__global__ void GaussianBlurKernelRow(
    float *d_data,
    float *out,
    int w,
    int h,
    int ksize,
    int pitch
)
{

    __shared__ float s[ROW_BLOCK_DIM_Y][ROW_BLOCK_DIM_X*(ROW_UNROLL_STEPS+ROW_HALO_STEP*2)];

    //base shared memory coordinate
    int baseX = (blockIdx.x*ROW_UNROLL_STEPS-ROW_HALO_STEP)*blockDim.x + threadIdx.x;
    int baseY = blockIdx.y*blockDim.y+threadIdx.y;

    //the data basing shared memory coordinate
    d_data += baseY * pitch + baseX;
    out    += baseY * pitch + baseX;

    //Load main data
#pragma unroll
    for(int i = ROW_HALO_STEP;i<ROW_UNROLL_STEPS+ROW_HALO_STEP;i++)
        s[threadIdx.y][threadIdx.x+ i * ROW_BLOCK_DIM_X] = (baseX + ROW_BLOCK_DIM_X * i < w ) ? d_data[ROW_BLOCK_DIM_X * i] : 0;


    //Load left halo
    //left halo exist when this is threads in the imgae patch.
#pragma unroll
    for (int i = 0; i < ROW_HALO_STEP; i++)
    {
        s[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = (baseX >= -i * ROW_BLOCK_DIM_X ) ? d_data[i * ROW_BLOCK_DIM_X] : 0;
    }


    //Load right halo
    //left halo exist when this is threads in the imgae patch.
#pragma unroll
    for (int i = ROW_HALO_STEP + ROW_UNROLL_STEPS; i < ROW_HALO_STEP + ROW_UNROLL_STEPS + ROW_HALO_STEP; i++)
    {
        s[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = (w - baseX > i * ROW_BLOCK_DIM_X) ? d_data[i * ROW_BLOCK_DIM_X] : 0;
    }

    __syncthreads();


    int b = (ksize -1) /2;
    for (int i = ROW_HALO_STEP; i < ROW_HALO_STEP + ROW_UNROLL_STEPS; i++)
    {
        float sum = 0;

#pragma unroll
        for (int j = -b; j <= b; j++)
        {
            sum += coeffGaussKernel[b-j] * s[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X + j];
        }

        out[i * ROW_BLOCK_DIM_X] = sum;
    }

    //old version
//     int b = (ksize -1) /2;
//     if(x>=b && x<w-b && y>=0 && y<h){
//        #pragma unroll
//        float sum = 0;
//        for(int i = -b;i<=b;i++){
//            sum += d_data[y*pitch+x+i]*coeffGaussKernel[i+b];
//        }
//        out[y*pitch+x] = sum;
//     }
}
///////////////////////////////////
/// \brief GaussianBlurKernelCol
/// \param d_data
/// \param out
/// \param w
/// \param h
/// \param ksize
/// \param pitch
/// There is a different with row that the col has not the pitch which could make sure the \
/// all thereds in image aera.
/// Reference the cuda-sample 'convolutionSeparable'
/// The boundary is set 0 which is different from OpenCV.The reason I simplify the boundary is \
/// that the description of the sift need not the boundary of the image which will be filter out.
/// The minimum y size is 64(COLUMNS_BLOCKDIM_Y*COLUMNS_RESULT_STEPS)
//////////////////////////////////

#define   COLUMNS_BLOCKDIM_X 32
#define   COLUMNS_BLOCKDIM_Y 16
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 2

__global__ void GaussianBlurKernelCol(
    float *d_data,
    float *out,
    int w,
    int h,
    int ksize,
    int pitch
)
{

    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_data += baseY * pitch + baseX;
    out    += baseY * pitch + baseX;
    int b = (ksize -1) /2;

    //fill the shared memory not consider the upper halo,so it limit the minimum y size is 64(COLUMNS_BLOCKDIM_Y*COLUMNS_RESULT_STEPS)
    if(baseY + (COLUMNS_RESULT_STEPS+COLUMNS_HALO_STEPS)*COLUMNS_BLOCKDIM_Y >= h && baseY + COLUMNS_HALO_STEPS*COLUMNS_BLOCKDIM_Y < h)
    {


        //Main data and lower halo
#pragma unroll
        for (int i = COLUMNS_HALO_STEPS; i <  COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS*2 ; i++)
        {
            s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < h) ? d_data[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
        }

        //Upper halo
#pragma unroll

        for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
        {
            s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_data[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
        }

        __syncthreads();
        for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        {
            float sum = 0;
#pragma unroll

            for (int j = -b ; j <= b; j++)
            {
                sum += coeffGaussKernel[b - j]* s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
            }

            if(baseY + i * COLUMNS_BLOCKDIM_Y < h) {
                out[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
            }
        }

        return;
    }


    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_data[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_data[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (h - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_data[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();


#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -b ; j <= b; j++)
        {
            sum += coeffGaussKernel[b - j]* s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        out[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }

#if 0
    if(y>=b && y<h-b && x>=0 && x<w){
        #pragma unroll
        for(int i = 0;i<ksize;i++){
            if(i<b){
                out[y*pitch+x] += d_data[(y-b+i)*pitch+x]*coeffGaussKernel[i];
            }
            else{
                out[y*pitch+x] += d_data[(y+i-b)*pitch+x]*coeffGaussKernel[i];
            }
        }
    }
#else
//    if(y>=b && y<h-b && x>=0 && x<w){
//       #pragma unroll
//       float sum = 0;
//       for(int i = -b;i<=b;i++){
//           sum += d_data[(y+i)*pitch+x]*coeffGaussKernel[i+b];
//       }
//       out[y*pitch+x] = sum;
//    }
#endif

}

__global__ void GaussianBlurKernelRTex(float *out,int w,int h,int ksize)
{

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int b = (ksize -1) /2;

    if(x>=b && x<w-b && y>=0 && y<h){
        #pragma unroll
        for(int i = 0;i<ksize;i++){
            if(i<b){
                out[y*w+x] += tex1Dfetch(texRef,y*w+x-b+i)*coeffGaussKernel[i];
            }
            else{
                out[y*w+x] += tex1Dfetch(texRef,y*w+x+i-b)*coeffGaussKernel[i];
            }
        }
    }

}
__global__ void differenceImg(float *d_Octave0,float *d_Octave1,float *d_diffOctave,int pitch,int height){

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int index = y * pitch + x;
    if(y<height)
        d_diffOctave[index] = (d_Octave1[index] - d_Octave0[index]);

}

__global__ void findScaleSpaceExtrema(float *prev,float *img,float *next,float *d_point,int width ,int pitch ,int height)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

//    if(x<1 || y<1 || x>=width-1 || y>=height-1)
//        return;

    const int BLOCKDIMX = 32;
    const int BLOCKDIMY = 8;

    __shared__ float Mem0[BLOCKDIMY+2][BLOCKDIMX+2];
    __shared__ float Mem1[BLOCKDIMY+2][BLOCKDIMX+2];
    __shared__ float Mem2[BLOCKDIMY+2][BLOCKDIMX+2];

    //the count of the extrema points in current block;
    __shared__ unsigned int cnt;
    //points storage in shared memory
    __shared__ unsigned short points[96];


//    float *ptr0 = prev[y * pitch + x];
//    float *ptr1 = img[y * pitch + x];
//    float *ptr2 = next[y * pitch + x];
    prev += ( y-1 ) * pitch + x - 1;
    img  += ( y-1 ) * pitch + x - 1;
    next += ( y-1 ) * pitch + x - 1;

    Mem0[ty][tx] = (x<0||y<0)? 0:prev[0];
    Mem1[ty][tx] = (x<0||y<0)? 0:img[0];
    Mem2[ty][tx] = (x<0||y<0)? 0:next[0];

//    Mem1[ty][32] = -400;
//    Mem1[8][tx] = -400;
//    Mem1[8][32] = -400;
    //prev[0] = 250;

    if(tx == 0 && ty == 0){
        #pragma unroll
        for(int i = BLOCKDIMY;i<BLOCKDIMY + 2;i++)
            #pragma unroll
            for(int j = 0;j<BLOCKDIMX+2;j++){
                Mem0[i][j] = (x<width||y<height)? prev[i*pitch + j]:0;
                Mem1[i][j] = (x<width||y<height)? img[i*pitch + j]:0;
                Mem2[i][j] = (x<width||y<height)? next[i*pitch + j]:0;
            }
        #pragma unroll
        for(int i = 0;i<BLOCKDIMY;i++)
            #pragma unroll
            for(int j = BLOCKDIMX;j<2+BLOCKDIMX;j++){
                Mem0[i][j] = (x<width||y<height)? prev[i*pitch + j]:0;
                Mem1[i][j] = (x<width||y<height)? img[i*pitch + j]:0;
                Mem2[i][j] = (x<width||y<height)? next[i*pitch + j]:0;
            }
      cnt = 0;
      //for points count synchronism
    }
    __syncthreads();
    prev += pitch + 1;
    img += pitch + 1;
    next += pitch + 1;

//    prev[0] = Mem0[ty+1][tx+1] + 200;
//    img[0] =  Mem1[ty+1][tx+1] + 200;
//    next[0] = Mem2[ty+1][tx+1] + 200 ;
    //next[0] = Mem2[ty+1][tx+1]*50 ;
    const int threshold = int(0.5 * 0.04 / 3 * 255);

    float val = img[0];
    int c = 0;
    int step = pitch;
    float *currptr = img;
    float *nextptr = next;
    float *prevptr = prev;
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



        int pos = atomicInc(&cnt, 31);
        points[3*pos+0] = x;
        points[3*pos+1] = y;
        //points[3*pos+2] = scale;


        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
        idx = (idx>=2000 ? 2000-1 : idx);
        d_point[idx*2] = x;
        d_point[idx*2+1] = y;


        printf("cnt : %d , x = %d , y = %d,asd: %f \n",idx,x,y,d_point[idx*2]);
    }


}

__device__ void addpoint(){

}


//////
/// \brief findScaleSpaceExtrema
/// \param d_point
/// \param s
/// \param width
/// \param pitch
/// \param height
/// \param threshold
/// \param nOctaveLayers
/// \param maxNum
////////////
/// s is the index in dog

__global__ void findScaleSpaceExtrema(float *d_point,int s, int width ,int pitch ,int height,const int threshold,const int nOctaveLayers,const int maxNum){

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
    if( abs(val) > threshold &&
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

            if( abs(Vs) < 0.5f && abs(Vx) < 0.5f && abs(Vy) < 0.5f )
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
            if( abs( contr ) * nOctaveLayers < 0.04 )
                return;

            // principal curvatures are computed using the trace and det of Hessian
            float tr = dxx + dyy;
            float det = dxx*dyy-dxy*dxy;

            if( det <= 0 || tr*tr*10 >= (10 + 1)*(10 + 1)*det )
                return;
        }

        layer = s - o*(nOctaveLayers+2);
#if 1
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
#else
        //addpoint;
        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
        idx = (idx>maxNum ? maxNum-1 : idx);
        d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
        d_point[idx*KEYPOINTS_SIZE+3] = size;
        d_point[idx*KEYPOINTS_SIZE+4] = abs(contr);
#endif
        /******************calOrientationHist*****************/
        {
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
            //float hist[n];

            //the procress of all point range, a square space.
            int k, len = (radius*2+1)*(radius*2+1);
            //garuss smooth's coefficient
            float expf_scale = -1.f/(2.f * sigma_ori * sigma_ori);
            //n = 36
            float *buf = new float[len*4 + n+4 + n];
            //the buf is a memory storage the temporary data.
            //The frist len is the Mag('fu zhi')and X,second len is the Y,third len is the Ori,
            //the forth is gauss weight(len+2)
            //the temphist is(n + 2).
            float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
            //gradient direction histogarm
            float* temphist = W + len + 2,*hist = temphist+n+2;

            for( int i = 0; i < n; i++ )
                temphist[i] = 0.f;

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

                    X[k] = dx;
                    Y[k] = dy;
                    //Wight not multiply 1/pi,because the compute of oritentation
                    //only need the relative wight.
                    W[k] = __expf((i*i + j*j)*expf_scale);
                    Ori[k] = atan2f(dy,dx);
                    Mag[k] = sqrtf(dy*dy+dx*dx);

                    //cvRound((ori/pi+180)/360*36)
                    float tembin = __fdividef(__fdividef(Ori[k]*180,CV_PI),360/n);
                    int bin = tembin > 0 ? tembin + 0.5:tembin - 0.5;
                    if( bin >= n )
                        bin -= n;
                    if( bin < 0 )
                        bin += n;
                    temphist[bin] += W[k]*Mag[k];
//                    if(k == 0)
//                    printf("temphist[%d]: %f , Mag[k] : %f , Y[k] :  %f \n",bin,temphist[bin],Mag[k],Y[k]);
                    //printf("bin : %d , Mag[k]: %f, W[k]: %f ,temphist[bin] %f \n",bin,Mag[k],W[k],temphist[bin]);
                    //printf("Mag[k] : %f,  X[k] :  %f , Y[k] :  %f \n",Mag[k],X[k],Y[k]);
                    k++;
                }
            }
            //printf("pixel : %f \n",currptr[0]);
//            for(int i = 0;i<len;i++)
//            {
//                Ori[i] = atan2f(Y[i],X[i]);
//                Mag[i] = sqrtf(Y[i]*Y[i]+X[i]*X[i]);
//            }

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
//                    kpt.angle = 360.f - (float)((360.f/n) * bin);
//                    if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
//                        kpt.angle = 0.f;

                    //addpoint;
#if 1
                    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                    idx = (idx>maxNum ? maxNum-1 : idx);
                    d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
                    d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
                    d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
                    d_point[idx*KEYPOINTS_SIZE+3] = size;
                    d_point[idx*KEYPOINTS_SIZE+4] = abs(contr);
                    d_point[idx*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
//                    kpt.pt.x = (c + xc) * (1 << octv);
//                    kpt.pt.y = (r + xr) * (1 << octv);
//                    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
//                    //why '*2'
//                    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
//                    kpt.response = std::abs(contr);

#else
#endif
                }
            }

            delete []buf;
        }//orientation
    }//extrema
}

__device__ void calcDescriptors(float* currptr,int x,int y,float scl_octv,int pitch,int width,int height,float ori,float* d_decriptor,int index)
{
    //description array
    //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    //x,y,360-angle,scl,d,n
    //static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    //x y is the x y in prymid image
    //scl_octv is the related scale in octave
    //x,y,scl_octv has been calculated above
    /******************calcDescriptor*****************/

    int d = SIFT_DESCR_WIDTH,n = SIFT_DESCR_HIST_BINS;
    ori = 360.f - ori;
    if(std::abs(ori - 360.f) < FLT_EPSILON)
        ori = 0.f;
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    //n=8
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //3*scale,normalized 3*scale to 1
    float hist_width = SIFT_DESCR_SCL_FCTR * scl_octv;
    int radius = int(hist_width * 1.4142135623730951f * (d + 1) * 0.5f+0.5);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = min(radius, (int) sqrt(((double) width)*width + ((double) height)*height));
    cos_t /= hist_width;
    sin_t /= hist_width;
    //len 为特征点邻域区域内像素的数量，histlen 为直方图的数量，即特征矢量的长度，实际应为d×d×n，之所以每个变量
    //又加上了2，是因为要为圆周循环留出一定的内存空间
    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    float dst[SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS];
    int rows = height, cols = width;
    float *buf = new float[len*6 + histlen];
    //Memory arrangment:
    //      Mag
    // X     Y    Ori    W   RBin  CBin  hist
    // -_____-_____-_____-_____-_____-_____-__
    //
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

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
    //2.all x,y in bins coordinary(a relate coordinary)
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
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
               // W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;

                W[k] = __expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                Ori[k] = atan2f(dy,dx);
                Mag[k] = sqrtf(dy*dy+dx*dx);
                k++;
            }
        }
    k = 0;
    for( ; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 =  rbin - (int)rbin ;
        int c0 =  cbin - (int)cbin;
        int o0 =  obin - (int)obin;
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

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
    }

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
        float val = min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/max(sqrtf(nrm2), FLT_EPSILON);

    for( ; k < len; k++ )
    {
        //dst[k] = (uchar)(dst[k]*nrm2);
        d_decriptor[index*len + k] = (uchar)(dst[k]*nrm2);
    }
    delete []buf;

}


__global__ void calcPerOctaveLayers(float *d_point,float* d_decriptor,int s, int width ,int pitch ,int height,const int threshold,const int nOctaveLayers,const int maxNum){

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
    if( abs(val) > threshold &&
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

            if( abs(Vs) < 0.5f && abs(Vx) < 0.5f && abs(Vy) < 0.5f )
                break;

            //get nearest intger for next iteration
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
            if( abs( contr ) * nOctaveLayers < 0.04 )
                return;

            // principal curvatures are computed using the trace and det of Hessian
            float tr = dxx + dyy;
            float det = dxx*dyy-dxy*dxy;

            if( det <= 0 || tr*tr*10 >= (10 + 1)*(10 + 1)*det )
                return;
        }

        layer = s - o*(nOctaveLayers+2);
#if 1
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
#else
        //addpoint;
        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
        idx = (idx>maxNum ? maxNum-1 : idx);
        d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
        d_point[idx*KEYPOINTS_SIZE+3] = size;
        d_point[idx*KEYPOINTS_SIZE+4] = abs(contr);
#endif
        float ori = 0.;
        float scl_octv = size*0.5f/(1 << o);
        unsigned int idx_arr[2];
        float ori_arr[2];
        int num_idx = 0;
        /******************calOrientationHist*****************/
        {
            //currptr is the current dog image where the current extrema point in.
            //x,y,s is the current location in dog images.
            //Note: s is the absolutely scale location and the 'laryer' is the /
            //relatively location in the octave which range is 1~3.

            //The orientation is compute in gausspyrmid,so the currptr renew:
            currptr = pgpyr[o*(nOctaveLayers+3) + layer]+y*pitch+x;
            //simga*2^s/S,the simga the simga relative to the octave.

            float omax;
            float sigma_ori = SIFT_ORI_SIG_FCTR * scl_octv;
            //'+0.5' for rounding because scl_octv>0
            int radius = SIFT_ORI_RADIUS * scl_octv+0.5,n = SIFT_ORI_HIST_BINS;
            //float hist[n];

            //the procress of all point range, a square space.
            int k, len = (radius*2+1)*(radius*2+1);
            //garuss smooth's coefficient
            float expf_scale = -1.f/(2.f * sigma_ori * sigma_ori);
            //n = 36
            float *buf = new float[len*4 + n+4 + n];
            //the buf is a memory storage the temporary data.
            //The frist len is the Mag('fu zhi')and X,second len is the Y,third len is the Ori,
            //the forth is gauss weight(len+2)
            //the temphist is(n + 2).
            float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
            //gradient direction histogarm
            float* temphist = W + len + 2,*hist = temphist+n+2;

            for( int i = 0; i < n; i++ )
                temphist[i] = 0.f;

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

                    X[k] = dx;
                    Y[k] = dy;
                    //Wight not multiply 1/pi,because the compute of oritentation
                    //only need the relative wight.
                    W[k] = __expf((i*i + j*j)*expf_scale);
                    Ori[k] = atan2f(dy,dx);
                    Mag[k] = sqrtf(dy*dy+dx*dx);

                    //cvRound((ori/pi+180)/360*36)
                    float tembin = __fdividef(__fdividef(Ori[k]*180,CV_PI),360/n);
                    int bin = tembin > 0 ? tembin + 0.5:tembin - 0.5;
                    if( bin >= n )
                        bin -= n;
                    if( bin < 0 )
                        bin += n;
                    temphist[bin] += W[k]*Mag[k];
//                    if(k == 0)
//                    printf("temphist[%d]: %f , Mag[k] : %f , Y[k] :  %f \n",bin,temphist[bin],Mag[k],Y[k]);
                    //printf("bin : %d , Mag[k]: %f, W[k]: %f ,temphist[bin] %f \n",bin,Mag[k],W[k],temphist[bin]);
                    //printf("Mag[k] : %f,  X[k] :  %f , Y[k] :  %f \n",Mag[k],X[k],Y[k]);
                    k++;
                }
            }
            //printf("pixel : %f \n",currptr[0]);
//            for(int i = 0;i<len;i++)
//            {
//                Ori[i] = atan2f(Y[i],X[i]);
//                Mag[i] = sqrtf(Y[i]*Y[i]+X[i]*X[i]);
//            }

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
//                    kpt.angle = 360.f - (float)((360.f/n) * bin);
//                    if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
//                        kpt.angle = 0.f;

                    //addpoint;
#if 1
                    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                    idx = (idx>maxNum ? maxNum-1 : idx);
                    d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
                    d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
                    d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
                    d_point[idx*KEYPOINTS_SIZE+3] = size;
                    d_point[idx*KEYPOINTS_SIZE+4] = abs(contr);
                    ori = 360.f - (float)((360.f/n) * bin);
                    if(abs(ori - 360.f) < FLT_EPSILON)
                        ori = 0.f;
                    d_point[idx*KEYPOINTS_SIZE+5] = ori;
//                    kpt.pt.x = (c + xc) * (1 << octv);
//                    kpt.pt.y = (r + xr) * (1 << octv);
//                    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
//                    //why '*2'
//                    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
//                    kpt.response = std::abs(contr);
                    ori_arr[num_idx] = ori;
                    idx_arr[num_idx] = idx;
                    num_idx++;
#else
#endif
                }
            }
            delete []buf;
        }//orientation
        num_idx = min(num_idx,2);
        for(int i = 0;i<num_idx;i++)
            calcDescriptors(currptr,x,y,scl_octv,pitch,width,height,ori_arr[num_idx],d_decriptor,idx_arr[num_idx]);
    }//extrema
}


__global__ void findScaleSpaceExtrema_gpu(float *d_point,int s, int width ,int pitch ,int height,const int threshold,const int nOctaveLayers,const int maxNum){

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
    if( abs(val) > threshold &&
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

            if( abs(Vs) < 0.5f && abs(Vx) < 0.5f && abs(Vy) < 0.5f )
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
            if( abs( contr ) * nOctaveLayers < 0.04 )
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
        d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
        d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
        d_point[idx*KEYPOINTS_SIZE+3] = size;
        d_point[idx*KEYPOINTS_SIZE+4] = abs(contr);
        d_point[idx*KEYPOINTS_SIZE+6] = s;
        d_point[idx*KEYPOINTS_SIZE+7] = x;
        d_point[idx*KEYPOINTS_SIZE+8] = y;
        //temsize+=size*0.5f/(1 << o)*SIFT_ORI_RADIUS+0.5;

        float scl_octv = size*0.5f/(1 << o);
        //'+0.5' for rounding because scl_octv>0
        int radius = SIFT_ORI_RADIUS * scl_octv+0.5;
        //the procress of all point range, a square space.
        int len = (radius*2+1)*(radius*2+1);
        //int temBuffSize = len*4+2*SIFT_ORI_HIST_BINS+2;
        atomicMax(&temsize,len);
    }
}

__global__ void calcOrientationHist_gpu(float *d_point,float* temdata,const int buffSize,const int pointsNum,const int maxNum,const int nOctaveLayers)
{
    //int x = blockIdx.x*blockDim.x+threadIdx.x;
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;
#define SHAREMEMORY
#ifdef SHAREMEMORY
    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

    __syncthreads();
    float size =s_point[threadIdx.x*KEYPOINTS_SIZE+3];
    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
#else
    float size =d_point[pointIndex*KEYPOINTS_SIZE+3];
    int s = d_point[pointIndex*KEYPOINTS_SIZE+6];
    int x = d_point[pointIndex*KEYPOINTS_SIZE+7];
    int y = d_point[pointIndex*KEYPOINTS_SIZE+8];
#endif


    int o = s/(nOctaveLayers+2);
    int layer = s - o*(nOctaveLayers+2);

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

            X[k] = dx;
            Y[k] = dy;
            //Wight not multiply 1/pi,because the compute of oritentation
            //only need the relative wight.
            W[k] = __expf((i*i + j*j)*expf_scale);
            Ori[k] = atan2f(dy,dx);
            Mag[k] = sqrtf(dy*dy+dx*dx);

            //cvRound((ori/pi+180)/360*36)
            float tembin = __fdividef(__fdividef(Ori[k]*180,CV_PI),360/n);
            int bin = tembin > 0 ? tembin + 0.5:tembin - 0.5;
            if( bin >= n )
                bin -= n;
            if( bin < 0 )
                bin += n;
            temphist[bin] += W[k]*Mag[k];
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
#ifdef SHAREMEMORY
            if(hist[j] == omax)
                d_point[pointIndex*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            else{
                //addpoint;
                unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                idx = (idx>maxNum ? maxNum-1 : idx);
                d_point[idx*KEYPOINTS_SIZE]   = s_point[threadIdx.x*KEYPOINTS_SIZE];
                d_point[idx*KEYPOINTS_SIZE+1] = s_point[threadIdx.x*KEYPOINTS_SIZE+1];
                d_point[idx*KEYPOINTS_SIZE+2] = s_point[threadIdx.x*KEYPOINTS_SIZE+2];
                d_point[idx*KEYPOINTS_SIZE+3] = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
                d_point[idx*KEYPOINTS_SIZE+4] = s_point[threadIdx.x*KEYPOINTS_SIZE+4];
                d_point[idx*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
                d_point[idx*KEYPOINTS_SIZE+6] = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
                d_point[idx*KEYPOINTS_SIZE+7] = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
                d_point[idx*KEYPOINTS_SIZE+8] = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
            }
#else
            if(hist[j] == omax)
                d_point[pointIndex*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            else{
                //addpoint;
                unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                idx = (idx>maxNum ? maxNum-1 : idx);
                d_point[idx*KEYPOINTS_SIZE]   = d_point[pointIndex*KEYPOINTS_SIZE];
                d_point[idx*KEYPOINTS_SIZE+1] = d_point[pointIndex*KEYPOINTS_SIZE+1];
                d_point[idx*KEYPOINTS_SIZE+2] = d_point[pointIndex*KEYPOINTS_SIZE+2];
                d_point[idx*KEYPOINTS_SIZE+3] = d_point[pointIndex*KEYPOINTS_SIZE+3];
                d_point[idx*KEYPOINTS_SIZE+4] = d_point[pointIndex*KEYPOINTS_SIZE+4];
                d_point[idx*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            }
#endif
        }
    }

    delete []hists;
}

__global__ void calcOrientationHist_gpu1(float *d_point,float* temdata,const int buffSize,const int pointsNum,const int maxNum,const int nOctaveLayers)
{
    //int x = blockIdx.x*blockDim.x+threadIdx.x;
    int pointIndex = blockIdx.x*blockDim.x+threadIdx.x;
    if(pointIndex>=pointsNum)
        return;
#define SHAREMEMORY
#ifdef SHAREMEMORY
    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

    __syncthreads();
    float size =s_point[threadIdx.x*KEYPOINTS_SIZE+3];
    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
#else
    float size =d_point[pointIndex*KEYPOINTS_SIZE+3];
    int s = d_point[pointIndex*KEYPOINTS_SIZE+6];
    int x = d_point[pointIndex*KEYPOINTS_SIZE+7];
    int y = d_point[pointIndex*KEYPOINTS_SIZE+8];
#endif


    int o = s/(nOctaveLayers+2);
    int layer = s - o*(nOctaveLayers+2);

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
#ifdef SHAREMEMORY
            if(hist[j] == omax)
                d_point[pointIndex*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            else{
                //addpoint;
                unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                idx = (idx>maxNum ? maxNum-1 : idx);
                d_point[idx*KEYPOINTS_SIZE]   = s_point[threadIdx.x*KEYPOINTS_SIZE];
                d_point[idx*KEYPOINTS_SIZE+1] = s_point[threadIdx.x*KEYPOINTS_SIZE+1];
                d_point[idx*KEYPOINTS_SIZE+2] = s_point[threadIdx.x*KEYPOINTS_SIZE+2];
                d_point[idx*KEYPOINTS_SIZE+3] = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
                d_point[idx*KEYPOINTS_SIZE+4] = s_point[threadIdx.x*KEYPOINTS_SIZE+4];
                d_point[idx*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
                d_point[idx*KEYPOINTS_SIZE+6] = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
                d_point[idx*KEYPOINTS_SIZE+7] = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
                d_point[idx*KEYPOINTS_SIZE+8] = s_point[threadIdx.x*KEYPOINTS_SIZE+8];

            }
#else
            if(hist[j] == omax)
                d_point[pointIndex*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            else{
                //addpoint;
                unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                idx = (idx>maxNum ? maxNum-1 : idx);
                d_point[idx*KEYPOINTS_SIZE]   = d_point[pointIndex*KEYPOINTS_SIZE];
                d_point[idx*KEYPOINTS_SIZE+1] = d_point[pointIndex*KEYPOINTS_SIZE+1];
                d_point[idx*KEYPOINTS_SIZE+2] = d_point[pointIndex*KEYPOINTS_SIZE+2];
                d_point[idx*KEYPOINTS_SIZE+3] = d_point[pointIndex*KEYPOINTS_SIZE+3];
                d_point[idx*KEYPOINTS_SIZE+4] = d_point[pointIndex*KEYPOINTS_SIZE+4];
                d_point[idx*KEYPOINTS_SIZE+5] = 360.f - (float)((360.f/n) * bin);
            }
#endif
        }
    }

    delete []hists;
}


__global__ void calcSIFTDescriptor_gpu(float *d_point,float* d_decriptor,int pointsNum,int  nOctaveLayers)
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

#define SHAREMEMORY
#ifdef SHAREMEMORYa
    __shared__ float s_point[BLOCK_SIZE_ONE_DIM*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE]   =d_point[pointIndex*KEYPOINTS_SIZE];
    s_point[threadIdx.x*KEYPOINTS_SIZE+1] =d_point[pointIndex*KEYPOINTS_SIZE+1];
    s_point[threadIdx.x*KEYPOINTS_SIZE+2] =d_point[pointIndex*KEYPOINTS_SIZE+2];
    s_point[threadIdx.x*KEYPOINTS_SIZE+3] =d_point[pointIndex*KEYPOINTS_SIZE+3];
    s_point[threadIdx.x*KEYPOINTS_SIZE+4] =d_point[pointIndex*KEYPOINTS_SIZE+4];
    s_point[threadIdx.x*KEYPOINTS_SIZE+5] =d_point[pointIndex*KEYPOINTS_SIZE+5];
    s_point[threadIdx.x*KEYPOINTS_SIZE+6] =d_point[pointIndex*KEYPOINTS_SIZE+6];
    s_point[threadIdx.x*KEYPOINTS_SIZE+7] =d_point[pointIndex*KEYPOINTS_SIZE+7];
    s_point[threadIdx.x*KEYPOINTS_SIZE+8] =d_point[pointIndex*KEYPOINTS_SIZE+8];

    __syncthreads();
    float size = s_point[threadIdx.x*KEYPOINTS_SIZE+3];
    float ori = s_point[threadIdx.x*KEYPOINTS_SIZE+5];
    int s = s_point[threadIdx.x*KEYPOINTS_SIZE+6];
    int x = s_point[threadIdx.x*KEYPOINTS_SIZE+7];
    int y = s_point[threadIdx.x*KEYPOINTS_SIZE+8];
#else
    float size =d_point[pointIndex*KEYPOINTS_SIZE+3];
    float ori = d_point[pointIndex*KEYPOINTS_SIZE+5];
    int s = d_point[pointIndex*KEYPOINTS_SIZE+6];
    int x = d_point[pointIndex*KEYPOINTS_SIZE+7];
    int y = d_point[pointIndex*KEYPOINTS_SIZE+8];
#endif

    int o = s/(nOctaveLayers+2);
    int layer = s - o*(nOctaveLayers+2);
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
    radius = min(radius, (int) sqrt(((double) width)*width + ((double) height)*height));
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
    //2.all x,y in bins coordinary(a relate coordinary)
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
                ok = atan2f(dy,dx);
                ok = (ok*180/CV_PI);
                ok = ok<0? ok+360:ok;
                mk = sqrtf(dy*dy+dx*dx);

                //float rbin = RBin[k], cbin = CBin[k];
                float obin = (ok - ori)*bins_per_rad;
                float mag = mk*wk;


                int r0 =  floor(rbin);
                int c0 =  floor(cbin);
                int o0 =  floor(obin);

                rbin -= r0;
                cbin -= c0;
                obin -= o0;

                if( o0 < 0 )
                    o0 += n;
                if( o0 >= n )
                    o0 -= n;

//                if(x == 1936 && y ==744 ){
//                    printf("k: %d,rbin: %f cbin: %f obin: %f mag: %f ok: %f\n",k,rbin,cbin,obin,mag,ok);
//                }


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
        float val = min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    __syncthreads();
    nrm2 = SIFT_INT_DESCR_FCTR/max(sqrtf(nrm2), FLT_EPSILON);
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



// Scale down thread block width
#define SCALEDOWN_W   160
// Scale down thread block height
#define SCALEDOWN_H    16
__constant__ float d_Kernel1[5];

__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W+4];
  __shared__ float brow[5*(SCALEDOWN_W/2)];
  __shared__ int yRead[SCALEDOWN_H+4];
  __shared__ int yWrite[SCALEDOWN_H+4];
  #define dx2 (SCALEDOWN_W/2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0*dx2;
  const int tx1 = tx + 1*dx2;
  const int tx2 = tx + 2*dx2;
  const int tx3 = tx + 3*dx2;
  const int tx4 = tx + 4*dx2;
  const int xStart = blockIdx.x*SCALEDOWN_W;
  const int yStart = blockIdx.y*SCALEDOWN_H;
  const int xWrite = xStart/2 + tx;
  const float *k = d_Kernel1;
  if (tx<SCALEDOWN_H+4) {
    int y = yStart + tx - 1;
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = y*pitch;
    yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - 2;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);
  for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
    inrow[tx] = d_Data[yRead[dy+0] + xRead];
    __syncthreads();
    if (tx<dx2)
      brow[tx0] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
    __syncthreads();
    if (tx<dx2 && dy>=4 && !(dy&1))
      d_Result[yWrite[dy+0] + xWrite] = k[2]*brow[tx2] + k[0]*(brow[tx0]+brow[tx4]) + k[1]*(brow[tx1]+brow[tx3]);
    if (dy<(SCALEDOWN_H+3)) {
      inrow[tx] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<dx2)
    brow[tx1] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=3 && (dy&1))
    d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]);
    }
    if (dy<(SCALEDOWN_H+2)) {
      inrow[tx] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<dx2)
    brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=2 && !(dy&1))
    d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]);
    }
    if (dy<(SCALEDOWN_H+1)) {
      inrow[tx] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<dx2)
    brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=1 && (dy&1))
    d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]);
    }
    if (dy<SCALEDOWN_H) {
      inrow[tx] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<dx2)
    brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && !(dy&1))
    d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]);
    }
    __syncthreads();
  }
}

__global__ void test()
{

//    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
//    printf("cnt : %d \n",d_PointCounter[0]);
}

void testDiffimage(float *d_Octave0,float *d_Octave1,float *d_diffOctave,int pitch,int height){
    dim3 Block(32,8);
    dim3 Grid(iDivUp(pitch,Block.x),iDivUp(height,Block.y));

    differenceImg<<<Grid,Block>>>(d_Octave0,d_Octave1,d_diffOctave,pitch,height);
    safeCall(cudaDeviceSynchronize());


}

//input cudaImage and output cudaImage which d_data has been smooth
void cuGaussianBlur(CudaImage &cuImg,float sigma)
{

    assert(sigma>0);
    int kernelSize = 0;
    //sigma = sqrtf(sigma * sigma - 0.5 * 0.5 * 4);

    //why the
    //ksize.width = cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    kernelSize = cvRound(sigma*4*2 + 1)|1;
    assert( kernelSize < 32*2+1 );

    Mat kx;
    kx = getGaussianKernel(kernelSize,sigma,CV_32F);

    CHECK(cudaMemcpyToSymbol(coeffGaussKernel,(float*)kx.data,sizeof(float)*kernelSize));

    dim3 BlockRow(ROW_BLOCK_DIM_X,ROW_BLOCK_DIM_Y);
    dim3 GridRow(iDivUp(cuImg.pitch,BlockRow.x*ROW_UNROLL_STEPS),iDivUp(cuImg.height,BlockRow.y));

    float *tmp_data,*tmp_data1;

    size_t pitch;
    safeCall(cudaMallocPitch((void**)&tmp_data, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    GaussianBlurKernelRow<<<GridRow,BlockRow>>>(cuImg.d_data,tmp_data,cuImg.width,cuImg.height,kernelSize,cuImg.pitch);
    safeCall(cudaDeviceSynchronize());

    safeCall(cudaMallocPitch((void**)&tmp_data1, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    dim3 BlockCol(COLUMNS_BLOCKDIM_X,COLUMNS_BLOCKDIM_Y);
    dim3 GridCol(iDivUp(cuImg.pitch,BlockCol.x),iDivUp(cuImg.height,BlockCol.y*COLUMNS_RESULT_STEPS));

    GaussianBlurKernelCol<<<GridCol,BlockCol>>>(tmp_data,tmp_data1,cuImg.width,cuImg.height,kernelSize,cuImg.pitch);
    safeCall(cudaDeviceSynchronize());

    /*device data has not copy to host yet*/
    safeCall(cudaMemcpy2D(cuImg.d_data,cuImg.pitch*sizeof(float),tmp_data1,cuImg.pitch*sizeof(float),cuImg.width*sizeof(float),(size_t) cuImg.height,cudaMemcpyDeviceToDevice));


#if 0
    Mat dis(cuImg.height,cuImg.width,CV_32F);
    safeCall(cudaMemcpy2D(dis.data,cuImg.width*sizeof(float),tmp_data1,cuImg.pitch*sizeof(float),cuImg.width*sizeof(float),(size_t) cuImg.height,cudaMemcpyDeviceToHost));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 0);
    cvNamedWindow("ss",CV_WINDOW_NORMAL);
    imshow("ss",gray);
    waitKey();
#endif

    cudaFree(tmp_data);
    cudaFree(tmp_data1);

#if MESSAGE == 0
    std::cout<<kernelSize<<std::endl;
    for(int i= 0 ;i<kx.rows;i++)
        for(int j = 0;j<kx.cols;j++){
            std::cout<<kx.at<float>(i,j)<<std::endl;
        }
#endif


}

void createInitialImage(const Mat &src, CudaImage &base, float sigma,bool doubleImageSize)
{
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
        cuGaussianBlur(base,sig_diff);
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        base.Allocate(width,height,iAlignUp(width, 128),false,NULL,(float*)gray_fpt.data);
        base.Download();
        cuGaussianBlur(base,sig_diff);
        //GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
    }


}

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
//  double a = 0.6;
//  float h_Kernel[5] = {1.0/4 - a/2.0, 1.0/4, a, 1.0/4, 1.0/4 - a/2.0};
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;
  safeCall(cudaMemcpyToSymbol(d_Kernel1, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

void buildGaussianPyramid(CudaImage& base, std::vector<CudaImage>& pyr, int nOctaves){
    //the vector of sigma per octave
    std::vector<double> sig(nOctaveLayers + 3);
    //init the size of the pyramid images which is nOctave*nLayer
    pyr.resize(nOctaves*(nOctaveLayers + 3));

#define USE_SEPARATION_MEMORY
#ifdef USE_SEPARATION_MEMORY
    //allocate separation memory
    int w = base.width;
    int h = base.height;
    for( int o = 0; o < nOctaves; o++ )
    {
        if(o != 0){
            w /= 2;
            h /= 2;
        }
        for( int i = 0; i < nOctaveLayers + 3; i++ ){
            pyr[o*(nOctaveLayers + 3) + i].Allocate(w,h,iAlignUp(w, 128),false);
        }
    }
#else
    //optimization points which allocate a big memory
    int w = base.width;
    int h = base.height;
    int pyrDataSize = 0;
    for( int o = 0; o < nOctaves; o++ )
    {
        if(o != 0){
            w /= 2;
            h /= 2;
        }
        int p = iAlignUp(w,128);
        pyrDataSize += (nOctaveLayers+3)*p*h;
    }
    float* d_pyrData = NULL;
    cudaMalloc(&d_pyrData,pyrDataSize*sizeof(float));
    //size_t pitch;
    //safeCall(cudaMallocPitch((void **)&d_pyrData, &pitch, (size_t)4096, (pyrDataSize+4095)/4096));
    //safeCall(cudaMallocPitch((void **)&d_pyrData, &pitch, (size_t)4096, (pyrDataSize+4095)/4096*sizeof(float)));
    int memLocation = 0;
    w = base.width;
    h = base.height;
    for( int o = 0; o < nOctaves; o++ )
    {
        if(o != 0){
            w /= 2;
            h /= 2;
        }
        for( int i = 0; i < nOctaveLayers + 3; i++ ){
            int p = iAlignUp(w,128);
            pyr[o*(nOctaveLayers + 3) + i].Allocate(w,h,p,false,d_pyrData+memLocation);
            //because the d_pyrData is the type of float so the offset of the
            //pointer is p*h rather than p*h*sizeof(float)
            memLocation += p*h;
        }
    }

//    CudaImage& src = pyr[0*(nOctaveLayers + 3)];
//    CudaImage& dst = pyr[0*(nOctaveLayers + 3)+1];
//    dst.copyDevice(src,1);

#endif
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
            CudaImage& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 ){
                dst.copyDevice(base,1);
#ifdef SHOW_GAUSSIANPYRAMID
                CudaImage &src = dst;
                Mat gray,show;
                show.create(src.height,src.width,CV_32F);
                safeCall(cudaMemcpy2D(show.data,src.width*sizeof(float),src.d_data,src.pitch*sizeof(float),src.width*sizeof(float),(size_t) src.height,cudaMemcpyDeviceToHost));
                show.convertTo(gray,DataType<uchar>::type, 1, 0);
                cvNamedWindow("ss",CV_WINDOW_NORMAL);
                imshow("ss",gray);
                waitKey(0);
#endif
            }
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                CudaImage& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                ScaleDown(dst,src,0.5);
            }
            else
            {
                CudaImage& src = pyr[o*(nOctaveLayers + 3) + i-1];
                dst.copyDevice(src,1);
                cuGaussianBlur(dst,sig[i]);
#ifdef SHOW_GAUSSIANPYRAMID
                Mat gray,show;
                show.create(dst.height,dst.width,CV_32F);
                safeCall(cudaMemcpy2D(show.data,src.width*sizeof(float),dst.d_data,src.pitch*sizeof(float),src.width*sizeof(float),(size_t) src.height,cudaMemcpyDeviceToHost));
                show.convertTo(gray,DataType<uchar>::type, 1, 0);
                cvNamedWindow("ss",CV_WINDOW_NORMAL);
                imshow("ss",gray);
                waitKey(0);
#endif
            }
        }
    }
}

//could use cuda stream
void buildDoGPyramid( std::vector<CudaImage>& gpyr, std::vector<CudaImage>& dogpyr )
{
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );


    //could use cuda stream
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers + 2;i++){
            CudaImage& prev = gpyr[o*(nOctaveLayers + 3)+i];
            CudaImage& next = gpyr[o*(nOctaveLayers + 3)+i+1];
            CudaImage& diff = dogpyr[o*(nOctaveLayers + 2)+i];
            diff.Allocate(prev.width,prev.height,prev.pitch,false);
            dim3 Block(32,8);
            dim3 Grid(iDivUp(diff.pitch,Block.x),iDivUp(diff.height,Block.y));
            differenceImg<<<Grid,Block>>>(prev.d_data,next.d_data,diff.d_data,diff.pitch,diff.height);
            safeCall(cudaDeviceSynchronize());
#ifdef SHOW_DOGPYRAMID
            Mat gray,show;
            show.create(diff.height,diff.width,CV_32F);
            safeCall(cudaMemcpy2D(show.data,diff.width*sizeof(float),diff.d_data,diff.pitch*sizeof(float),diff.width*sizeof(float),(size_t) diff.height,cudaMemcpyDeviceToHost));
            show.convertTo(gray,DataType<uchar>::type, 30, 200);
            cvNamedWindow("ss",CV_WINDOW_NORMAL);
            imshow("ss",gray);
            waitKey(0);
#endif
        }
    }


}
int getMaxDescriptorBufSize(int len){
    //get the max scl_oct
    int radius_ori = (sqrt(len)-1)/2;
    //int radius = SIFT_ORI_RADIUS * scl_octv+0.5;
    float maxScl_oct = (radius_ori + 1)/SIFT_ORI_RADIUS;
    int radius_des = int((SIFT_DESCR_SCL_FCTR * maxScl_oct * 1.4142135623730951f * (SIFT_DESCR_WIDTH + 1) * 0.5f)+0.5);
    return int((radius_des*2+1)*(radius_des*2+1));
}
void findScaleSpaceExtrema(std::vector<CudaImage>& gpyr, std::vector<CudaImage>& dogpyr, std::vector<KeyPoint>& keypointss, Mat &descriptors){

    float* d_keypoints;
    float* h_keypoints;

    int totPts = 0;
    safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
    cudaMalloc(&d_keypoints,sizeof(float)*maxPoints*KEYPOINTS_SIZE);

    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    //std::cout<<"my threshold = "<<threshold<<std::endl;
#ifdef FIND_DOGERRORTEST
#else
    float **h_pd = new float*[dogpyr.size()];
#endif
    for(int i = 0;i<dogpyr.size();i++)
        h_pd[i] = dogpyr[i].d_data;
    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*dogpyr.size()));


    float **h_gpyr = new float*[gpyr.size()];
    for(int i = 0;i<gpyr.size();i++)
        h_gpyr[i] = gpyr[i].d_data;
    safeCall(cudaMemcpyToSymbol(pgpyr, h_gpyr, sizeof(float *)*gpyr.size()));

    //for every OctaveLayers which number is o*3
#if 0
    //combine findextrema and oritentation
    dim3 Block(32,8);
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers;i++){
            int index = o*(nOctaveLayers+2)+i+1;
            dim3 Grid(iDivUp(dogpyr[index].pitch,Block.x),iDivUp(dogpyr[index].height,Block.y));
            findScaleSpaceExtrema<<<Grid,Block>>>(d_keypoints,index,dogpyr[index].width,dogpyr[index].pitch,dogpyr[index].height,threshold,nOctaveLayers,maxPoints);
            //calcPerOctaveLayers<<<Grid,Block>>>(d_keypoints,d_decriptor,index,dogpyr[index].width,dogpyr[index].pitch,dogpyr[index].height,threshold,nOctaveLayers,maxPoints);
            safeCall(cudaDeviceSynchronize());
        }
    }

#else
    int temDataSize = 0;
    safeCall(cudaMemcpyToSymbol(temsize, &temDataSize, sizeof(int)));

    dim3 Block(32,8);
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers;i++){
            int index = o*(nOctaveLayers+2)+i+1;
            dim3 Grid(iDivUp(dogpyr[index].pitch,Block.x),iDivUp(dogpyr[index].height,Block.y));
            findScaleSpaceExtrema_gpu<<<Grid,Block>>>(d_keypoints,index,dogpyr[index].width,dogpyr[index].pitch,dogpyr[index].height,threshold,nOctaveLayers,maxPoints);
            safeCall(cudaDeviceSynchronize());
        }
    }

    int num0 = 0;
    safeCall(cudaMemcpyFromSymbol(&num0, d_PointCounter, sizeof(int)));
    num0 = (num0>maxPoints)? maxPoints:num0;
    printf("cuda sift kepoints num : %d \n",num0);
    int* oIndex = new int[33];
    for(int i =0;i<nOctaves;i++){
        int index = i*(nOctaveLayers+2);
        oIndex[i*3] = dogpyr[index].width;
        oIndex[i*3+1] = dogpyr[index].height;
        oIndex[i*3+2] = dogpyr[index].pitch;
    }
    safeCall(cudaMemcpyToSymbol(d_oIndex, oIndex, sizeof(int)*33));

//    int* d_oIndex;
//    cudaMalloc(&d_oIndex,sizeof(int)*nOctaves*3);
//    cudaMemcpy(d_oIndex,oIndex,sizeof(int)*nOctaves*3,cudaMemcpyHostToDevice);

    float* temData;
    safeCall(cudaMemcpyFromSymbol(&temDataSize, temsize, sizeof(int)));
    //4 is the 4 len buf
    int buffSize = temDataSize*4;
    safeCall(cudaMalloc(&temData,sizeof(float)*num0*buffSize));
    //std::cout<<"buffSize:"<<buffSize<<std::endl;

    int grid =iDivUp(num0,BLOCK_SIZE_ONE_DIM);
    //use the global memory
    //calcOrientationHist_gpu<<<grid,BLOCK_SIZE_ONE_DIM>>>(d_keypoints,temData,buffSize,num0,maxPoints,nOctaveLayers);
    calcOrientationHist_gpu1<<<grid,BLOCK_SIZE_ONE_DIM>>>(d_keypoints,temData,buffSize,num0,maxPoints,nOctaveLayers);
    safeCall( cudaGetLastError() );
    safeCall(cudaDeviceSynchronize());
    cudaFree(temData);

    int num1 = 0;
    safeCall(cudaMemcpyFromSymbol(&num1, d_PointCounter, sizeof(int)));
    num1 = (num1>maxPoints)? maxPoints:num1;
    printf("cuda sift kepoints num : %d \n",num1);

    //alloc for d_decriptor
    float* d_descriptor;
    int despriptorSize = SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
    cudaMalloc(&d_descriptor,sizeof(float)*num1*despriptorSize);

    grid =iDivUp(num1,BLOCK_SIZE_ONE_DIM);
    calcSIFTDescriptor_gpu<<<grid,BLOCK_SIZE_ONE_DIM>>>(d_keypoints,d_descriptor,num1,nOctaveLayers);
    safeCall( cudaGetLastError() );
    safeCall(cudaDeviceSynchronize());


    float *h_descriptor;
    h_descriptor = (float *)malloc(num1*despriptorSize*sizeof(float));
    safeCall(cudaMemcpy(h_descriptor,d_descriptor,num1*despriptorSize*sizeof(float),cudaMemcpyDeviceToHost));

    descriptors.create(num1,128,CV_32FC1);
    safeCall(cudaMemcpy((float*)descriptors.data,d_descriptor,num1*128*sizeof(float),cudaMemcpyDeviceToHost));


#endif

    int num = 0;
    safeCall(cudaMemcpyFromSymbol(&num, d_PointCounter, sizeof(int)));
    num = (num>maxPoints)? maxPoints:num;
    printf("cuda sift kepoints num : %d \n",num);

    h_keypoints = (float *)malloc(num*KEYPOINTS_SIZE*sizeof(float));
    safeCall(cudaMemcpy(h_keypoints,d_keypoints,num*KEYPOINTS_SIZE*sizeof(float),cudaMemcpyDeviceToHost));



    cudaFree(d_keypoints);
    cudaFree(d_descriptor);

#ifdef SHOW_KEYPOINT

    //std::vector<KeyPoint> keypointss;
    keypointss.resize(num);
    for(int i = 0;i<keypointss.size();++i)
    {
        keypointss[i].pt.x =  h_keypoints[i*KEYPOINTS_SIZE];
        keypointss[i].pt.y =  h_keypoints[i*KEYPOINTS_SIZE+1];
        keypointss[i].octave =  h_keypoints[i*KEYPOINTS_SIZE+2];
        keypointss[i].size =  h_keypoints[i*KEYPOINTS_SIZE+3];
        keypointss[i].response =  h_keypoints[i*KEYPOINTS_SIZE+4];
        keypointss[i].angle =  h_keypoints[i*KEYPOINTS_SIZE+5];
    }

//    KeyPointsFilter::removeDuplicatedSorted( keypointss );
//    printf("my sift kepoints num after clear : %d \n",keypointss.size());


#ifdef NODOUBLEIMAGE
#else
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
#endif




//    Mat kepoint;
////    CudaImage &img = gpyr[0];
////    Mat img_1(img.height,img.width,CV_32F);
////    safeCall(cudaMemcpy2D(img_1.data,img.width*sizeof(float),gpyr[0].d_data,gpyr[0].pitch*sizeof(float),gpyr[0].width*sizeof(float),(size_t) gpyr[0].height,cudaMemcpyDeviceToHost));
//    //char *a ="../data/100_7101.JPG";
//    //char *a ="../data/img2.ppm";
//    //char *a ="../data/100_7101.JPG";
//    char *a ="../data/road.png";
//    Mat img_1 = imread(a);
//    Mat gray;
//    img_1.convertTo(gray,DataType<uchar>::type, 1, 0);
//    drawKeypoints(gray,keypointss,kepoint,cv::Scalar::all(-1),4);


//    cvNamedWindow("extract_my",CV_WINDOW_NORMAL);
//    imshow("extract_my", kepoint);
//    waitKey(0);

//    for(int i = 0;i < keypointss.size();i++)
//        std::cout<<keypointss[i].pt.x<<" ";
//    std::cout<<std::endl;
#ifdef COMPARE_VALUE
    sort(keypointss.begin(),keypointss.end(),sortx);
    int unique_nums;
    unique_nums = std::unique(keypointss.begin(),keypointss.end(),uniquex) - keypointss.begin();
    for(int i = 0;i < unique_nums;i++)
        std::cout<<keypointss[i].response<<" ";
    std::cout<<unique_nums<<std::endl;
#endif
#endif
    free(h_keypoints);
    free(h_descriptor);
}

void calcDescriptors(std::vector<CudaImage>& gpyr,float* d_keypoints){

}

void displayOctave(std::vector<CudaImage> &Octave)
{
    Mat display;
    int width = Octave[0].width;
    int height = Octave[0].height*Octave.size();
    display.create(height,width,CV_32F);

//    for(int i = 0 ;  i<Octave.size(); i++){
//        safeCall(cudaMemcpy2D(display.data+width*Octave[0].height*sizeof(float)*i,Octave[0].width*sizeof(float),Octave[0].d_data,Octave[0].pitch*sizeof(float),Octave[0].width*sizeof(float),(size_t) Octave[0].height,cudaMemcpyDeviceToHost));
//    }
    for(int i = 0 ;  i<Octave.size(); i++){
        safeCall(cudaMemcpy2D(display.data+Octave[i].width*Octave[i].height*i*sizeof(float),Octave[i].width*sizeof(float),Octave[i].d_data,Octave[i].pitch*sizeof(float),Octave[i].width*sizeof(float),(size_t) Octave[i].height,cudaMemcpyDeviceToHost));
    }
    Mat gray;
    display.convertTo(gray,DataType<uchar>::type, 1, 0);
    cvNamedWindow("a",CV_WINDOW_NORMAL);
    imshow("a",gray);
    waitKey(0);
}

void disMatf(char* name,CudaImage &img){
    Mat dis(img.height,img.width,CV_32F);
    safeCall(cudaMemcpy2D(dis.data,img.width*sizeof(float),img.d_data,img.pitch*sizeof(float),img.width*sizeof(float),(size_t) img.height,cudaMemcpyDeviceToHost));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 200);
    cvNamedWindow(name,CV_WINDOW_NORMAL);
    imshow(name,gray);

}

void computePerOctave(CudaImage& base, std::vector<double> &sig, int nOctaveLayers){

    std::vector<CudaImage> Octave;
    Octave.resize(nOctaveLayers + 3);
    Octave[0].copyDevice(base);
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        Octave[i].copyDevice(Octave[i-1]);
        cuGaussianBlur(Octave[i],sig[i]);
    }

    //displayOctave(Octave);

    std::vector<CudaImage> diffOctave;
    diffOctave.resize(nOctaveLayers+2);
    for(int i = 0;i<diffOctave.size();++i)
        diffOctave[i].Allocate(Octave[0].width,Octave[0].height,Octave[0].pitch,NULL,NULL);
//    float *d_data,pitch;
//    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*Octave[0].width*5), (size_t)Octave[0].height));

    dim3 Block(32,8);
    dim3 Grid(iDivUp(Octave[0].pitch,Block.x),iDivUp(Octave[0].height,Block.y));


    for(int i = 0;i<diffOctave.size();i++){
        differenceImg<<<Grid,Block>>>(Octave[i].d_data,Octave[i+1].d_data,diffOctave[i].d_data,Octave[0].pitch,Octave[0].height);
        safeCall(cudaDeviceSynchronize());
    }


#ifdef SHOW
    //displayOctave(diffOctave);
#endif

    ////////////////////
    /// findScaleSpaceExtrema
    ////////////////////


    int totPts = 0;
    safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
    float *d_point;
    cudaMalloc(&d_point,sizeof(float)*maxPoints*2);
    //for(int i = 0 ; i < diffOctave - 1;i++)
    int i = 2;
    //findScaleSpaceExtrema<<<Grid,Block>>>(diffOctave[i].d_data,diffOctave[i+1].d_data,diffOctave[i+2].d_data,d_point,Octave[0].width,Octave[0].pitch,Octave[0].height);
    //safeCall(cudaDeviceSynchronize());

//    float *p[3+2];
//    float d = 2;
//    float *s = &d;
//    p[0] = s;
//    std::cout<<*(p[0])<<"  "<< sizeof(float*) <<std::endl;

//    test<<<1,1>>>(p);

    float *h_pd[3+2];
    for(int i = 0;i<5;i++)
        h_pd[i] = diffOctave[i].d_data;
    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*5));

    int width = Octave[0].width;
    int pitch = Octave[0].pitch;
    int heigh = Octave[0].height;

    //findScaleSpaceExtrema<<<Grid,Block>>>(d_point,3,Octave[0].width,Octave[0].pitch,Octave[0].height);
    safeCall(cudaDeviceSynchronize());


#ifdef SHOW
    disMatf("prve",diffOctave[i]);
    disMatf("current",diffOctave[i+1]);
    disMatf("next",diffOctave[i+2]);
    waitKey(0);
#endif

//    test<<<2,23>>>();

    int num = 0;
    safeCall(cudaMemcpyFromSymbol(&num, d_PointCounter, sizeof(int)));
    num = (num>2000)? 2000:num;
    printf("width : %d , height : %d , num : %d \n",Octave[0].width,Octave[0].height,num);

    float *h_points;
    h_points = (float *)malloc(num*2*sizeof(float));
    //h_points = new float[num*2];
    safeCall(cudaMemcpy(h_points,d_point,num*2*sizeof(float),cudaMemcpyDeviceToHost));
    std::vector<KeyPoint> keypoints;
    keypoints.resize(num);
    for(int i = 0;i<keypoints.size();++i)
    {
        keypoints[i].pt.x =  h_points[i*2];
        keypoints[i].pt.y =  h_points[i*2+1];
    }



#ifdef SHOW
    Mat kepoint;
    CudaImage &img = diffOctave[i+1];
    Mat img_1(img.height,img.width,CV_32F);
    safeCall(cudaMemcpy2D(img_1.data,img.width*sizeof(float),diffOctave[i+1].d_data,img.pitch*sizeof(float),img.width*sizeof(float),(size_t) img.height,cudaMemcpyDeviceToHost));
    Mat gray;
    img_1.convertTo(gray,DataType<uchar>::type, 1, 200);
    drawKeypoints(gray,keypoints,kepoint);
    //char *a ="../data/road.png";
    //Mat img_1 = imread(a);
    //drawKeypoints(img_1,keypoints,kepoint);

    cvNamedWindow("extract_my",CV_WINDOW_NORMAL);
    imshow("extract_my", kepoint);
    waitKey(0);
#endif





}

/*disable*/
void disMatf(CudaImage &cuImg){
    Mat dis(cuImg.height,cuImg.width,CV_32F);

    for(int i = 0;i<dis.rows;i++)
    {
        float *p = dis.ptr<float>(i);
        for(int j = 0;j<dis.cols;j++){
            p[j] = cuImg.h_data[i*dis.cols+j];
            //std::cout<<p[j]<<" ";
        }
        //std::cout<<std::endl;
    }
    //memcpy(dis.data,cuImg.h_data,cuImg.width*cuImg.height*sizeof(float));
    Mat gray;
    dis.convertTo(gray,DataType<uchar>::type, 1, 0);

    cvNamedWindow("ff",CV_WINDOW_NORMAL);
    imshow("ff",gray);
    waitKey();
}
