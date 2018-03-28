#include "cusitf_function.h"

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


__global__ void foo()
{
    printf("CUDA!\n");
}


void useCUDA()
{

    foo<<<1,5>>>();
    CHECK(cudaDeviceSynchronize());

}
void cuGaussianBlur(InputArray src, OutputArray dst, Size ksize,
                    double sigmaX, double sigmaY = 0,
                    int borderType = BORDER_DEFAULT )
{

}
