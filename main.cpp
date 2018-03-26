#include <iostream>
#include <opencv2/highgui.hpp>


int main(int argc, char *argv[])
{

    cv::Mat img1 = cv::imread("../data/lena.png");
    cv::Mat img2(img1);

    //`Esc` drop out
    while(cvWaitKey(33) != 27)
    {
        cv::imshow("main1",img2);
    }

    cvDestroyAllWindows();
    img1.release();
    img2.release();

    return 0;
}
