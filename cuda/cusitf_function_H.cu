#include "cusitf_function_H.h"


#define MESSAGE 1


#define __MAXSIZECON 32*2+1
__constant__ float coeffGaussKernel[__MAXSIZECON];
__device__ unsigned int d_PointCounter[1];
//choose 55 suport 16384 pixel size image (log2(16384) - 3)*5
__device__ float *pd[55];
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
        int j = 0;
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


            //???   why is -pdx not pdx
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


//            if( std::abs(Vs) > 1.0f || std::abs(Vx) >1.0f || std::abs(Vy) > 1.0f ){
//             printf("*******  Vs : %f , Vx = %f , Vy = %f \n",Vs,Vx,Vy);
//             printf("*******intger  Vs : %d , Vx = %d , Vy = %d \n",int(Vs > 0 ? ( Vs + 0.5 ) : (Vs - 0.5)),int(Vx > 0 ? ( Vx + 0.5 ) : (Vx - 0.5)),int(Vy > 0 ? ( Vy + 0.5 ) : (Vy - 0.5)));
//            }

            int layer = s - o*(nOctaveLayers+2);

            //printf("scale : %d , laryer : %d , Vs: %f\n",s,layer,Vs);

            if( layer < 1 || layer > nOctaveLayers ||
                y < SIFT_IMG_BORDER || y >= height - SIFT_IMG_BORDER  ||
                x < SIFT_IMG_BORDER || x >= width - SIFT_IMG_BORDER )
                return;

        }//for
        if( j >= SIFT_MAX_INTERP_STEPS )
            return;

        //After the iterative,get the x,y,s,(Vx,Vy,Vs)(<0.5).

        {
//            currptr = pd[s]  +y*pitch+x;
//            prevptr = pd[s-1]+y*pitch+x;
//            nextptr = pd[s+1]+y*pitch+x;

//            dx = (currptr[1] - currptr[-1])*deriv_scale;
//            dy = (currptr[pitch] - currptr[-pitch])*deriv_scale;;
//            ds = (nextptr[0] - prevptr[0])*deriv_scale;

//            float v2 = currptr[0]*2;

//            dxx = (currptr[1] + currptr[-1] - v2)*second_deriv_scale;
//            dyy = (currptr[pitch] + currptr[-pitch] - v2)*second_deriv_scale;
//            dxy = (currptr[pitch+1] - currptr[1-pitch] -
//                         currptr[-1+pitch] + currptr[-pitch-1])*cross_deriv_scale;


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


        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
        idx = (idx>maxNum ? maxNum-1 : idx);
        d_point[idx*5] = (x + Vx)*(1 << o);
        d_point[idx*5+1] = (y + Vy)*(1 << o);

        //printf("cnt : %d , x = %f , y = %f \n",idx,d_point[idx*2],d_point[idx*2+1]);
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
                dst.copyDevice(base,0);
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
                dst.copyDevice(src,0);
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
            CudaImage& prev = gpyr[o*(nOctaveLayers + 2)+i+o];
            CudaImage& next = gpyr[o*(nOctaveLayers + 2)+i+1+o];
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

void findScaleSpaceExtrema(std::vector<CudaImage>& gpyr, std::vector<CudaImage>& dogpyr, float* keypoints){
    int totPts = 0;
    safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
    cudaMalloc(&keypoints,sizeof(float)*maxPoints*KeyPoints_size);


    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    float **h_pd = new float*[dogpyr.size()];
    for(int i = 0;i<dogpyr.size();i++)
        h_pd[i] = dogpyr[i].d_data;
    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*dogpyr.size()));

    //std::cout<<dogpyr.size()<<std::endl;
//    float* h_pd[30];
//    for(int i = 0;i<dogpyr.size();i++)
//        h_pd[i] = dogpyr[i].d_data;
//    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*30));



    dim3 Block(32,8);
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    for(int o = 0;o<nOctaves;o++){
        for(int i = 0;i<nOctaveLayers;i++){
            int index = o*(nOctaveLayers+2)+i+1;
            dim3 Grid(iDivUp(dogpyr[index].pitch,Block.x),iDivUp(dogpyr[index].height,Block.y));
            findScaleSpaceExtrema<<<Grid,Block>>>(keypoints,index,dogpyr[index].width,dogpyr[index].pitch,dogpyr[index].height,threshold,nOctaveLayers,maxPoints);
            safeCall(cudaDeviceSynchronize());
        }
    }




    //float *h_pd[5];
//    float **h_pd = new float*[5];
//    for(int i = 0;i<5;i++)
//        h_pd[i] = dogpyr[i].d_data;
//    safeCall(cudaMemcpyToSymbol(pd, h_pd, sizeof(float *)*5));


//    for(int i = 1;i<4;i++){
//        dim3 Block(32,8);
//        dim3 Grid(iDivUp(dogpyr[i].pitch,Block.x),iDivUp(dogpyr[i].height,Block.y));
//        findScaleSpaceExtrema<<<Grid,Block>>>(keypoints,i,dogpyr[i].width,dogpyr[i].pitch,dogpyr[i].height);
//        safeCall(cudaDeviceSynchronize());
//    }
#ifdef SHOW_KEYPOINT
    int num = 0;
    safeCall(cudaMemcpyFromSymbol(&num, d_PointCounter, sizeof(int)));
    num = (num>maxPoints)? maxPoints:num;
    printf("num : %d \n",num);

    float *h_points;
    h_points = (float *)malloc(num*KeyPoints_size*sizeof(float));
    safeCall(cudaMemcpy(h_points,keypoints,num*KeyPoints_size*sizeof(float),cudaMemcpyDeviceToHost));
    std::vector<KeyPoint> keypointss;
    keypointss.resize(num);
    for(int i = 0;i<keypointss.size();++i)
    {
        keypointss[i].pt.x =  h_points[i*KeyPoints_size];
        keypointss[i].pt.y =  h_points[i*KeyPoints_size+1];
    }

    Mat kepoint;
    CudaImage &img = gpyr[0];
    Mat img_1(img.height,img.width,CV_32F);
    safeCall(cudaMemcpy2D(img_1.data,img.width*sizeof(float),gpyr[0].d_data,gpyr[0].pitch*sizeof(float),gpyr[0].width*sizeof(float),(size_t) gpyr[0].height,cudaMemcpyDeviceToHost));
    Mat gray;
    img_1.convertTo(gray,DataType<uchar>::type, 1, 0);
    drawKeypoints(gray,keypointss,kepoint);
    //char *a ="../data/road.png";
    //Mat img_1 = imread(a);
    //drawKeypoints(img_1,keypoints,kepoint);

    cvNamedWindow("extract_my",CV_WINDOW_NORMAL);
    imshow("extract_my", kepoint);
    waitKey(0);
#endif

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
    cudaMalloc(&d_point,sizeof(float)*2000*2);
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
