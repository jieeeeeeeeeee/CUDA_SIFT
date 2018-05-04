#ifndef CUSIFT_H
#define CUSIFT_H

#include "cuImage.h"



namespace cusift {

class cuSIFT{
public:
    cuSIFT();
    void detectAndCompute(Mat &src, std::vector<KeyPoint> keypoints);
private:
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;

};

}





#endif
