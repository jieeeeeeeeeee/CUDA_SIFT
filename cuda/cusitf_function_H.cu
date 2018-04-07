#include "cusitf_function_H.h"


#define MESSAGE 1


#define __MAXSIZECON 200
__constant__ float coeffGaussKernel[__MAXSIZECON];
texture<float, 1, cudaReadModeElementType> texRef;


#define __CUDA_ARCH__ 600

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
   // const int BLOCK_DIM_X = 32;

//for row
#define   BLOCK_DIM_X 32
#define   BLOCK_DIM_Y 8
//    const int BLOCK_DIM_Y = 8;
#define UNROLL_STEPS 2
#define HALO_STEP 1

//for col
#define   COLUMNS_BLOCKDIM_X 32
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 2
#define   COLUMNS_HALO_STEPS 1


#else
    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 4;
#define UNROLL_STEPS 2
#define HALO_STEP 1

#define   COLUMNS_BLOCKDIM_X 32
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 1
#define   COLUMNS_HALO_STEPS 1


#endif


__global__ void GaussianBlurKernelRow(float *d_data,float *out,int w,int h,int ksize,int pitch){

    __shared__ float s[BLOCK_DIM_Y][BLOCK_DIM_X*(UNROLL_STEPS+HALO_STEP*2)];

    //base shared memory coordinate
    int baseX = (blockIdx.x*UNROLL_STEPS-HALO_STEP)*blockDim.x + threadIdx.x;
    int baseY = blockIdx.y*blockDim.y+threadIdx.y;

//    if(baseX + (UNROLL_STEPS+HALO_STEP)*BLOCK_DIM_X>w)
//        return;

    //the data basing shared memory coordinate
    d_data += baseY * pitch + baseX;
    out    += baseY * pitch + baseX;

    //Load main data
#pragma unroll
    for(int i = HALO_STEP;i<UNROLL_STEPS+HALO_STEP;i++)
        //s[threadIdx.y][threadIdx.x+ i * BLOCK_DIM_X] = d_data[BLOCK_DIM_X * i];
        s[threadIdx.y][threadIdx.x+ i * BLOCK_DIM_X] = (w <= baseY * pitch + baseX + BLOCK_DIM_X * i) ? d_data[BLOCK_DIM_X * i]:0;


    //Load left halo
#pragma unroll
    for (int i = 0; i < HALO_STEP; i++)
    {
        s[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = (baseX >= -i * BLOCK_DIM_X) ? d_data[i * BLOCK_DIM_X] : 0;
    }


    //Load right halo
#pragma unroll
    for (int i = HALO_STEP + UNROLL_STEPS; i < HALO_STEP + UNROLL_STEPS + HALO_STEP; i++)
    {
        s[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = (w - baseX > i * BLOCK_DIM_X) ? d_data[i * BLOCK_DIM_X] : 0;
    }

    __syncthreads();


    int b = (ksize -1) /2;
    for (int i = HALO_STEP; i < HALO_STEP + UNROLL_STEPS; i++)
    {
        float sum = 0;

#pragma unroll
        for (int j = -b; j <= b; j++)
        {
            sum += coeffGaussKernel[b-j] * s[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X + j];
        }

        out[i * BLOCK_DIM_X] = sum;
    }



#if 0 //not reduce the performance while has the 'if',because the branch influence the threads in a warp not in a thread
    int b = (ksize -1) /2;
    if(x>=b && x<w-b && y>=0 && y<h){
        #pragma unroll
        for(int i = 0;i<ksize;i++){
            if(i<b){
                out[y*pitch+x] += d_data[y*pitch+x-b+i]*coeffGaussKernel[i];
            }
            else{
                out[y*pitch+x] += d_data[y*pitch+x+i-b]*coeffGaussKernel[i];
            }
        }
    }
#else
//     int b = (ksize -1) /2;
//     if(x>=b && x<w-b && y>=0 && y<h){
//        #pragma unroll
//        float sum = 0;
//        for(int i = -b;i<=b;i++){
//            sum += d_data[y*pitch+x+i]*coeffGaussKernel[i+b];
//        }
//        out[y*pitch+x] = sum;
//     }
#endif
}
//__global__ void GaussianBlurKernelRow(float *d_data,float *out,int w,int h,int ksize,int pitch){

//    int x = blockIdx.x*blockDim.x+threadIdx.x;
//    int y = blockIdx.y*blockDim.y+threadIdx.y;

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
//    const int BLOCK_DIM_X = 32;
//    const int BLOCK_DIM_Y = 8;
//#else
//    const int BLOCK_DIM_X = 32;
//    const int BLOCK_DIM_Y = 4;
//    const int PATCH_PER_BLOCK = 4;
//    const int HALO_SIZE = 1;
//#endif
//    __shared__ float s[BLOCK_DIM_Y][BLOCK_DIM_X+HALO*2];


//    if(blockIdx.x > 0 && blockIdx.x < w/blockDim.x-1 ){
//        //main data
//        //for(int i = 0;i<BLOCK_DIM_Y;i++)
//        s[threadIdx.y][HALO+threadIdx.x] = d_data[y*pitch+x];

//        //Load left halo
//        if(threadIdx.x == 0)
//            #pragma unroll
//            for(int i = 0;i<HALO;i++)
//                s[threadIdx.y][threadIdx.x+i] = d_data[y*pitch+x-HALO+i];
//        //Load right halo
//        if(threadIdx.x == BLOCK_DIM_X - 1)
//            #pragma unroll
//            for(int i = 1;i<=HALO;i++)
//                s[threadIdx.y][threadIdx.x+HALO+i] = d_data[y*pitch+x+i];
//        __syncthreads();


//        int b = (ksize -1) /2;
//        #pragma unroll
//        for(int i = -b;i<=b;i++){
//            out[y*pitch+x] += s[threadIdx.y][threadIdx.x+HALO+i]*coeffGaussKernel[i+b];
//        }

//    }


//#if 0 //not reduce the performance while has the 'if',because the branch influence the threads in a warp not in a thread
//    int b = (ksize -1) /2;
//    if(x>=b && x<w-b && y>=0 && y<h){
//        #pragma unroll
//        for(int i = 0;i<ksize;i++){
//            if(i<b){
//                out[y*pitch+x] += d_data[y*pitch+x-b+i]*coeffGaussKernel[i];
//            }
//            else{
//                out[y*pitch+x] += d_data[y*pitch+x+i-b]*coeffGaussKernel[i];
//            }
//        }
//    }
//#else
////     int b = (ksize -1) /2;
////     if(x>=b && x<w-b && y>=0 && y<h){
////        #pragma unroll
////        for(int i = -b;i<=b;i++){
////            out[y*pitch+x] += d_data[y*pitch+x+i]*coeffGaussKernel[i+b];
////        }
////     }
//#endif
//}


__global__ void GaussianBlurKernelCol(float *d_data,float *out,int w,int h,int ksize,int pitch){

    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_data += baseY * pitch + baseX;
    out    += baseY * pitch + baseX;

//    if(baseY + (COLUMNS_RESULT_STEPS)*COLUMNS_BLOCKDIM_Y>h)
//        return;

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

    int b = (ksize -1) /2;
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -b ; j <= b; j++)
        {
            sum += coeffGaussKernel[b - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
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
//__global__ void GaussianBlurKernelCol(float *d_data,float *out,int w,int h,int ksize,int pitch){

//    int x = blockIdx.x*blockDim.x+threadIdx.x;
//    int y = blockIdx.y*blockDim.y+threadIdx.y;
////    if(x == 511)
////    {
////        #pragma unroll
////        for(int i =0;i<ksize;i++)
////            printf(" %d :  %f  \n",i,coeffGaussKernel[i]);
////    }
////    if(x<4&&y<4)
////    {
////        printf("d_data : %f ",d_data[x*w+y]);
////    }
//    //out[x*w+y] =d_data[x*w+y];
//    int b = (ksize -1) /2;
//#if 0
//    if(y>=b && y<h-b && x>=0 && x<w){
//        #pragma unroll
//        for(int i = 0;i<ksize;i++){
//            if(i<b){
//                out[y*pitch+x] += d_data[(y-b+i)*pitch+x]*coeffGaussKernel[i];
//            }
//            else{
//                out[y*pitch+x] += d_data[(y+i-b)*pitch+x]*coeffGaussKernel[i];
//            }
//        }
//    }
//#else
//    if(y>=b && y<h-b && x>=0 && x<w){
//       #pragma unroll
//       float sum = 0;
//       for(int i = -b;i<=b;i++){
//           sum += d_data[(y+i)*pitch+x]*coeffGaussKernel[i+b];
//       }
//       out[y*pitch+x] = sum;
//    }
//#endif

//}
__global__ void GaussianBlurKernelColShare(float *d_data,float *out,int w,int h,int ksize)
{
    int b = (ksize -1) /2;
    //__shared__ data[10+blockDim.x];

//    if(y>=b && y<h-b && x>=0 && x<w){


//    }

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






//void cuGaussianBlur(cuImage &cuImg,float sigma)
//{
//    //ksize.width = cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
//    //createsize
//    //getkernel
//    assert(sigma>0);
//    int kernelSize = 0;

//    //sigma = sqrtf(sigma * sigma - 0.5 * 0.5 * 4);

//    kernelSize = cvRound(sigma*4*2 + 1)|1;

//    Mat kx;
//    kx = getGaussianKernel(kernelSize,sigma,CV_32F);
////    for(int i = 0;i<kx.cols;i++)
////        coeffGaussKernel[i] = ((float*)kx.data)[i];

//    CHECK(cudaMemcpyToSymbol(coeffGaussKernel,(float*)kx.data,sizeof(float)*kernelSize));

//    dim3 Block(32,8);
//    dim3 Grid(iDivUp(cuImg.width,Block.x),iDivUp(cuImg.height,Block.y));
//    float *tmp_data,*tmp_data1;

//    safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));
//    GaussianBlurKernelRow<<<Grid,Block>>>(cuImg.d_data,tmp_data,cuImg.width,cuImg.height,kernelSize);
//    safeCall(cudaDeviceSynchronize());


//    safeCall(cudaMalloc(&tmp_data1,cuImg.width*cuImg.height*sizeof(float)));
//    GaussianBlurKernelCol<<<Grid,Block>>>(tmp_data,tmp_data1,cuImg.width,cuImg.height,kernelSize);
//    safeCall(cudaDeviceSynchronize());

//    safeCall(cudaMemcpy(cuImg.h_data,tmp_data1,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));

//    cudaFree(tmp_data);
//    cudaFree(tmp_data1);
//    disMatf(cuImg);

//      /*tex*/
////    CHECK(cudaBindTexture(NULL,texRef,cuImg.d_data,cuImg.width*cuImg.height*sizeof(float)));
////    dim3 Block(32,8);
////    dim3 Grid(iDivUp(cuImg.width,Block.x),iDivUp(cuImg.height,Block.y));
////    float *tmp_data,*tmp_data1;
////    safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));

////    GaussianBlurKernelRTex<<<Grid,Block>>>(tmp_data,cuImg.width,cuImg.height,kernelSize);

////    safeCall(cudaDeviceSynchronize());
////    safeCall(cudaMemcpy(cuImg.h_data,tmp_data,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));

////    disMatf(cuImg);
//    /*tex*/


//#if MESSAGE == 0
//    std::cout<<kernelSize<<std::endl;
//    for(int i= 0 ;i<kx.rows;i++)
//        for(int j = 0;j<kx.cols;j++){
//            std::cout<<kx.at<float>(i,j)<<std::endl;
//        }
//#endif


//    //size = cvRound(sigma*(sizeof(T) == CV_8U ? 3 : 4)*2 + 1)|1;

//}

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

    dim3 Block(BLOCK_DIM_X,BLOCK_DIM_Y);
    dim3 Grid(iDivUp(cuImg.pitch,Block.x*UNROLL_STEPS),iDivUp(cuImg.height,Block.y));

    float *tmp_data,*tmp_data1;

    size_t pitch;
    // safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));
    safeCall(cudaMallocPitch((void**)&tmp_data, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    GaussianBlurKernelRow<<<Grid,Block>>>(cuImg.d_data,tmp_data,cuImg.width,cuImg.height,kernelSize,cuImg.pitch);
    safeCall(cudaDeviceSynchronize());

    safeCall(cudaMallocPitch((void**)&tmp_data1, (size_t*) &pitch, (size_t) cuImg.width*sizeof(float),  (size_t) cuImg.height));

    dim3 BlockCol(COLUMNS_BLOCKDIM_X,COLUMNS_BLOCKDIM_Y);
    dim3 GridCol(iDivUp(cuImg.pitch,Block.x),iDivUp(cuImg.height,Block.y*COLUMNS_RESULT_STEPS));


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
