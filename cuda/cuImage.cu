#include "cuImage.h"



cuImage::cuImage()
    :width(0), height(0), d_data(NULL), h_data(NULL),t_data(NULL)
{

}

void cuImage::Allocate(
        int w, int h, int p, float *devMem, float *hostMem)
{
    width = w;
    height = h;
    pitch = p;
    d_data = devMem;
    h_data = hostMem;
//    safeCall(cudaMallocPitch((void **)&d_data, &pitch, (sizeof(float)*width), (sizeof(float)height));
//    pitch /= sizeof(float);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");

}
void cuImage::Allocate1D(
        int w, int h, float *hostMem)
{
    width = w;
    height = h;
    h_data = hostMem;

    safeCall(cudaMalloc(&d_data,width*height*sizeof(float)));
    if (d_data==NULL)
      printf("Failed to allocate device data\n");


}
void cuImage::AllocateMat1D(cv::Mat &src,bool withHost ){
    width = src.cols;
    height = src.rows;

    Mat gray, gray_fpt;
    if( src.channels() == 3 || src.channels() == 4 )
    {
        cvtColor(src, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt,DataType<float>::type, 1, 0);
    }
    else
        src.convertTo(gray_fpt,DataType<float>::type, 1, 0);

    h_data = (float*)gray_fpt.data;
    safeCall(cudaMalloc(&d_data,width*height*sizeof(float)));
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    safeCall(cudaMemcpy(d_data,h_data,width*height*sizeof(float),cudaMemcpyHostToDevice));
    if(withHost){
        hostIner = true;
        h_data = new float[height*width];
    }

}
cuImage::~cuImage()
{
    if (d_data!=NULL)
      safeCall(cudaFree(d_data));
    if (h_data!=NULL&&hostIner==true)
      free(h_data);
}

