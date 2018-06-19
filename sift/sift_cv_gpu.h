#ifndef SIFT_CV_GPU_H
#define SIFT_CV_GPU_H

#include "opencv2/features2d.hpp"


namespace cv
{
namespace xfeatures2d
{
namespace cvGpu {
class SIFT : public Feature2D
{
public:
    /**
    @param nfeatures The number of best features to retain. The features are ranked by their scores
    (measured in SIFT algorithm as the local contrast)

    @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
    number of octaves is computed automatically from the image resolution.

    @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
    (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

    @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
    is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
    filtered out (more features are retained).

    @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
    is captured with a weak camera with soft lenses, you might want to reduce the number.
     */
    static Ptr<SIFT> create( int nfeatures = 0, int nOctaveLayers = 3,
                                    double contrastThreshold = 0.04, double edgeThreshold = 10,
                                    double sigma = 1.6);
};
}
}
}
#endif
