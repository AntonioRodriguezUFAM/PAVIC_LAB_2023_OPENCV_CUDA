#include "opencvCPU.h"
#include "opencvGPU.h"

#include <iostream>
#include<opencv2/core/cuda.hpp>


using namespace std;
using namespace cv;
using namespace cuda;


int main() {

	cout << "CUDA Device Info: " << endl;
	
	try
	{
		// ... Contents of your main
		printCudaDeviceInfo(getDevice());
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl; // output exception message
	}

	cout << " Hello CUDA with Opencv" << endl;

	return 0;
}