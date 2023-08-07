#include "opencvCPU.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void opencvCPU::cpuSpeedTest() {
    VideoCapture cap(0);
    namedWindow("Image", WINDOW_AUTOSIZE);

    while (cap.isOpened()) {
        Mat image;
        bool isSuccess = cap.read(image);

        if (image.empty()) {
            cout << "Could not load Image!!" << endl;
            break;
        }

        auto start = getTickCount();
        Mat Img_Gray;
        Mat Img_Blur;

        cvtColor(image, Img_Gray, COLOR_BGR2GRAY);
        GaussianBlur(Img_Gray, Img_Blur, Size(9, 9), 3, 0);

       // GaussianBlur(Img_Gray, Img_Blur, Size(blurKernelSize), 3, 0);
        auto end = getTickCount();

        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;

        fps = 0.9 * old_fps + 0.1 * fps;

        putText(Img_Blur, " FPS: " + to_string(int(fps)), Point(50, 50), \
            FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
        old_fps = fps;

        if (getWindowProperty("Image", WND_PROP_AUTOSIZE) <= 0) {
            //imshow("Image", Img_Blur);
            imshow("Image", image);
            cv::waitKey(0);
        }
        else {
            break;
        }

        int k = waitKey(1);
        if (k == 27) {
            break;
        }
    }
}
