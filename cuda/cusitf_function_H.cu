#include "cusitf_function_H.h"


#define MESSAGE 1


#define __MAXSIZECON 64*2+1
__constant__ float coeffGaussKernel[__MAXSIZECON];
texture<float, 1, cudaReadModeElementType> texRef;

/***********
//This is an adjustable option which control the gaussKernel size. \
//when the kernel size less than 32*2+1 or kernel radius less than 32,the ROW_HALO_STEP set 1 \
//and the COLUMNS_HALO_STEPS set 2 will has a good performance.But when the kernel size is more \
//than 32 less than 64, the ROW_HALO_STEP should set 2 and the COLUMNS_HALO_STEPS should set 4
//The ROW_HALO_STEP will set 2 and the COLUMNS_HALO_STEPS will set 4 by default
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
/// The boundary is set 0.
/// If adjust the ROW_HALO_STEP 2,that is ok.
//////////////////////

#define   ROW_BLOCK_DIM_X 32
#define   ROW_BLOCK_DIM_Y 8
#define  ROW_UNROLL_STEPS 4
#define     ROW_HALO_STEP 2

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
        //s[threadIdx.y][threadIdx.x+ i * ROW_BLOCK_DIM_X] = d_data[ROW_BLOCK_DIM_X * i];
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
/// The boundary is set 0.
/// The minimum y size is 64(COLUMNS_BLOCKDIM_Y*COLUMNS_RESULT_STEPS)
//////////////////////////////////

#define   COLUMNS_BLOCKDIM_X 32
#define   COLUMNS_BLOCKDIM_Y 16
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 4

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


void cuGaussianBlur(CudaImage &cuImg,float sigma)
{
    //ksize.width = cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    //createsize
    //getkernel

    assert(sigma>0);
    assert(1);
    int kernelSize = 0;

    //sigma = sqrtf(sigma * sigma - 0.5 * 0.5 * 4);

    kernelSize = cvRound(sigma*4*2 + 1)|1;

    Mat kx;
    kx = getGaussianKernel(kernelSize,sigma,CV_32F);

    CHECK(cudaMemcpyToSymbol(coeffGaussKernel,(float*)kx.data,sizeof(float)*kernelSize));

    dim3 BlockRow(ROW_BLOCK_DIM_X,ROW_BLOCK_DIM_Y);
    dim3 GridRow(iDivUp(cuImg.pitch,BlockRow.x*ROW_UNROLL_STEPS),iDivUp(cuImg.height,BlockRow.y));

    float *tmp_data,*tmp_data1;

    size_t pitch;
    // safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));
    safeCall(cudaMallocPitch((void**)&tmp_data, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    GaussianBlurKernelRow<<<GridRow,BlockRow>>>(cuImg.d_data,tmp_data,cuImg.width,cuImg.height,kernelSize,cuImg.pitch);
    safeCall(cudaDeviceSynchronize());

    safeCall(cudaMallocPitch((void**)&tmp_data1, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    dim3 BlockCol(COLUMNS_BLOCKDIM_X,COLUMNS_BLOCKDIM_Y);
    dim3 GridCol(iDivUp(cuImg.pitch,BlockCol.x),iDivUp(cuImg.height,BlockCol.y*COLUMNS_RESULT_STEPS));


    //safeCall(cudaMalloc(&tmp_data1,cuImg.width*cuImg.height*sizeof(float)));
    GaussianBlurKernelCol<<<GridCol,BlockCol>>>(tmp_data,tmp_data1,cuImg.width,cuImg.height,kernelSize,cuImg.pitch);
    safeCall(cudaDeviceSynchronize());

    /*device data has not copy to host yet*/



#if 1
    Mat dis(cuImg.height,cuImg.width,CV_32F);
    //safeCall(cudaMemcpy(dis.data,tmp_data1,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(dis.data,cuImg.width*sizeof(float),tmp_data1,cuImg.pitch*sizeof(float),cuImg.width*sizeof(float),(size_t) cuImg.height,cudaMemcpyDeviceToHost));

//    Mat dis(cuImg.height,cuImg.pitch,CV_32F);
//    //safeCall(cudaMemcpy(dis.data,tmp_data1,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));
//    safeCall(cudaMemcpy2D(dis.data,cuImg.pitch*sizeof(float),tmp_data1,cuImg.pitch*sizeof(float),cuImg.width*sizeof(float),(size_t) cuImg.height,cudaMemcpyDeviceToHost));


    //    for(int i = 0;i<dis.rows;i++)
//    {
//        float *p = dis.ptr<float>(i);
//        for(int j = 0;j<dis.cols;j++){
//            p[j] = cuImg.h_data[i*dis.cols+j];
//            //std::cout<<p[j]<<" ";
//        }
//        //std::cout<<std::endl;
//    }
//    memcpy(dis.data,tmp_data1,cuImg.width*cuImg.height*sizeof(float));
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

    cvNamedWindow("ss",CV_WINDOW_NORMAL);
    imshow("ss",gray);
    waitKey();
}
