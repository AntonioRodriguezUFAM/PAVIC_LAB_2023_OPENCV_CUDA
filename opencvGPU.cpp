#include "opencvGPU.h"
#include<opencv2/core/cuda.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cuda;

void opencvGPU::gpuSpeedTest() {
	printCudaDeviceInfo(0);

}