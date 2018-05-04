#ifndef CUIMAGE_H
#define CUIMAGE_H

#include <stdlib.h>
#include "cuGlobal.h"


class cuImage{
public:
    cuImage();
    ~cuImage();

    //void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
    void Allocate(int width, int height, int pitch, float *devMem = NULL, float *hostMem = NULL);
    void Allocate1D(int width, int height, float *hostMem = NULL);
    void AllocateMat1D(cv::Mat &src, bool withHost=true);

    int width;
    int height;
    int channel;
    int pitch;

    float* d_data;
    float* h_data;
    float* t_data;


private:
    //float* h_data;
    bool hostIner;
};


namespace cusift {
class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
  void copyDevice(CudaImage &src);
  void copyDevice(CudaImage &src,bool haveDevice);

};

}


#endif
