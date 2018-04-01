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





#endif
