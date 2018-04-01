#ifndef CUGLOBAL_H
#define CUGLOBAL_H


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>





#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

using namespace cv;



//for grid
inline int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
inline int iDivDown(int a, int b) { return a/b; }
//for pitch
inline int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
inline int iAlignDown(int a, int b) { return a - a%b; }

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __safeThreadSync(const char *file, const int line)
{
  cudaError err = cudaThreadSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}


class TimerGPU {
public:
  cudaEvent_t start, stop;
  cudaStream_t stream;
  TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
  }
  ~TimerGPU() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  float read() {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};

class TimerCPU
{
  static const int bits = 10;
public:
  long long beg_clock;
  float freq;
  TimerCPU(float freq_) : freq(freq_) {   // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits) {
#ifdef WIN32
    return __rdtsc()/(1LL<<bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
    return ((long long)high<<(32-bits)) | ((long long)low>>bits);
#endif
  }
  float read() {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1<<bits)*Kcycles/freq/1e3f;
    return time;
  }
};

#endif
