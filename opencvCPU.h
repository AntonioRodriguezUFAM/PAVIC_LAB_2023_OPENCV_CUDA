#pragma once

class opencvCPU
{
public:
    double old_fps = 100;
    void cpuSpeedTest();

private:
    static const int blurKernelSize = 9;
};
