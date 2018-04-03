#include "cusitf_function_H.h"


#define MESSAGE 1



#define __MAXSIZECON 100
__constant__ float coeffGaussKernel[__MAXSIZECON];
texture<float, 1, cudaReadModeElementType> texRef;

__global__ void foo()
{
    printf("CUDA!\n");
}


__global__ void kernel(int size){
    #pragma unroll
    for(int i =0;i<size;i++)
        printf(" %d :  %f  \n",i,coeffGaussKernel[i]);
}
__global__ void GaussianBlurKernelRow(float *d_data,float *out,int w,int h,int ksize){

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int b = (ksize -1) /2;

    if(x>=b && x<w-b && y>=0 && y<h){
        #pragma unroll
        for(int i = 0;i<ksize;i++){
            if(i<b){
                out[y*w+x] += d_data[y*w+x-b+i]*coeffGaussKernel[i];
            }
            else{
                out[y*w+x] += d_data[y*w+x+i-b]*coeffGaussKernel[i];
            }
        }
    }

}

__global__ void GaussianBlurKernelCol(float *d_data,float *out,int w,int h,int ksize){

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
//    if(x == 511)
//    {
//        #pragma unroll
//        for(int i =0;i<ksize;i++)
//            printf(" %d :  %f  \n",i,coeffGaussKernel[i]);
//    }
//    if(x<4&&y<4)
//    {
//        printf("d_data : %f ",d_data[x*w+y]);
//    }
    //out[x*w+y] =d_data[x*w+y];
    int b = (ksize -1) /2;

    if(y>=b && y<h-b && x>=0 && x<w){
        #pragma unroll
        for(int i = 0;i<ksize;i++){
            if(i<b){
                out[y*w+x] += d_data[(y-b+i)*w+x]*coeffGaussKernel[i];
            }
            else{
                out[y*w+x] += d_data[(y+i-b)*w+x]*coeffGaussKernel[i];
            }
        }
    }
}

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
void useCUDA()
{

    foo<<<1,5>>>();
    CHECK(cudaDeviceSynchronize());

}



void cuGaussianBlur(cuImage &cuImg,float sigma)
{
    //ksize.width = cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    //createsize
    //getkernel
    assert(sigma>0);
    int kernelSize = 0;

    //sigma = sqrtf(sigma * sigma - 0.5 * 0.5 * 4);

    kernelSize = cvRound(sigma*4*2 + 1)|1;

    Mat kx;
    kx = getGaussianKernel(kernelSize,sigma,CV_32F);
//    for(int i = 0;i<kx.cols;i++)
//        coeffGaussKernel[i] = ((float*)kx.data)[i];

    CHECK(cudaMemcpyToSymbol(coeffGaussKernel,(float*)kx.data,sizeof(float)*kernelSize));

    dim3 Block(32,8);
    dim3 Grid(iDivUp(cuImg.width,Block.x),iDivUp(cuImg.height,Block.y));
    float *tmp_data,*tmp_data1;

    safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));
    GaussianBlurKernelRow<<<Grid,Block>>>(cuImg.d_data,tmp_data,cuImg.width,cuImg.height,kernelSize);
    safeCall(cudaDeviceSynchronize());


    safeCall(cudaMalloc(&tmp_data1,cuImg.width*cuImg.height*sizeof(float)));
    GaussianBlurKernelCol<<<Grid,Block>>>(tmp_data,tmp_data1,cuImg.width,cuImg.height,kernelSize);
    safeCall(cudaDeviceSynchronize());

    safeCall(cudaMemcpy(cuImg.h_data,tmp_data1,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));

    cudaFree(tmp_data);
    cudaFree(tmp_data1);
    disMatf(cuImg);

      /*tex*/
//    CHECK(cudaBindTexture(NULL,texRef,cuImg.d_data,cuImg.width*cuImg.height*sizeof(float)));
//    dim3 Block(32,8);
//    dim3 Grid(iDivUp(cuImg.width,Block.x),iDivUp(cuImg.height,Block.y));
//    float *tmp_data,*tmp_data1;
//    safeCall(cudaMalloc(&tmp_data,cuImg.width*cuImg.height*sizeof(float)));

//    GaussianBlurKernelRTex<<<Grid,Block>>>(tmp_data,cuImg.width,cuImg.height,kernelSize);

//    safeCall(cudaDeviceSynchronize());
//    safeCall(cudaMemcpy(cuImg.h_data,tmp_data,cuImg.width*cuImg.height*sizeof(float),cudaMemcpyDeviceToHost));

//    disMatf(cuImg);
    /*tex*/


#if MESSAGE == 0
    std::cout<<kernelSize<<std::endl;
    for(int i= 0 ;i<kx.rows;i++)
        for(int j = 0;j<kx.cols;j++){
            std::cout<<kx.at<float>(i,j)<<std::endl;
        }
#endif


    //size = cvRound(sigma*(sizeof(T) == CV_8U ? 3 : 4)*2 + 1)|1;

}
void disMatf(cuImage &cuImg){
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
