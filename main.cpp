#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "opencvGPU.h"
#include "opencvCPU.h"

using namespace std;

int main() {
 
    cout << "Hello CUDA with OpenCV" << endl;

    opencvCPU CPU;
    CPU.cpuSpeedTest();

    return 0;
}
