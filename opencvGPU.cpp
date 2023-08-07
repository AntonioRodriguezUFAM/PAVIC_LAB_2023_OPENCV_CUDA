#include "opencvGPU.h",
#include<opencv2/core/cuda.hpp>
#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;
using namespace cuda;

void opencvGPU::gpuSpeedTest() {
	int dispH1 = 800;
	int dispH1 = 600;

	VideoCapture cap(1);

	double old_fps = 100;



	printCudaDeviceInfo(0);

	namedWindow("ImageOutput", WINDOW_AUTOSIZE);

	/*Ptr<cuda::filter> gaussian_filter_9x9 = \
		cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(9, 9), 3, 0);

	while (cap.isOpened()) {
		



	}*/

}