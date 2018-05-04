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

namespace cusift {

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem==NULL) {
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL)
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
}

double CudaImage::Download()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL)
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::Readback()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::InitTexture()
{
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

//new operator
void CudaImage::copyDevice(CudaImage &src){
    width = src.width;
    height = src.height;
    pitch = src.pitch;
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    safeCall(cudaMemcpy2D(d_data, sizeof(float)*pitch, src.d_data, sizeof(float)*src.pitch, sizeof(float)*width, height, cudaMemcpyDeviceToDevice));
    d_internalAlloc = true;
}
void CudaImage::copyDevice(CudaImage &src,bool haveDevice){
    width = src.width;
    height = src.height;
    pitch = src.pitch;
    if(!haveDevice){
        safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
        d_internalAlloc = true;
    }
    pitch /= sizeof(float);
    safeCall(cudaMemcpy2D(d_data, sizeof(float)*pitch, src.d_data, sizeof(float)*src.pitch, sizeof(float)*width, height, cudaMemcpyDeviceToDevice));
}

}
