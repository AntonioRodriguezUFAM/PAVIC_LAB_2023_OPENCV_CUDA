//QuickDemo.h

# pragma once 
# include <opencv2/opencv.hpp> # include <iostream> # include <opencv2/features2d/features2d.hpp> # include <opencv2/dnn.hpp> # include <windows.h> # include "clock_.h " 






using  namespace cv;
using namespace cv::dnn;
using namespace std;

class  QuickDemo {
public:
	//SURF feature point detection void Sift (Mat &src1, Mat &src2) ;	


	   //Hue one-dimensional histogram 
	void Hue(Mat& src);

	//Hue-saturation two-dimensional histogram 
	void Hue_Saruration(Mat& src);

	//Watershed algorithm (bug) 
//void water_shed(Mat &src);	

	//Find contours 
	void find_contours(Mat& src);

	//Histogram equalization 
	void equal_hist(Mat& src);

	//Hough transform 
	void SHT(Mat& src);

	//scharr edge detection 
	void scharr_edge(Mat& src);

	//laplacian edge detection 
	void laplacian_edge(Mat& src);

	//sobel edge detection 
	void sobel_edge(Mat& src);

	//High-order canny edge detection 
	void canny_edge_plus(Mat& src);

	//canny edge detection 
	void canny_edge(Mat& src);

	//Image pyramid - down collection 
	void pyr_down(Mat& img);

	//Image pyramid - upward collection 
	void pyr_up(Mat& img);

	//Fill 
	void flood_fill(Mat& img);

	// Morphological summary 
	void morpho(Mat& img, int op);

	//morphological gradient 
	void morpho_gradient(Mat& img);

	//top hat 
	void top_hat(Mat& img);

	//Black hat 
	void black_hat(Mat& img);

	//opening operation 
	void op_opening();

	//Closing operation 
	void op_closing();

	//expansion 
	void img_dilate();

	//Erosion 
	void img_erode();

	//xml, yml file read and write 
	void file_storage();

	//ROI_AddImage 
	void ROI_AddImage();

	~QuickDemo();

	//Space color conversion 
	void colorSpace_Demo(Mat& image);

	//Mat creates image 
	void matCreation_Demo(Mat& image);

	//Image pixel read and write 
	void pixelVisit_Demo(Mat& image);

	//Image pixel arithmetic operation 
	void operators_Demo(Mat& image);

	//Scroll bar to adjust image brightness 
	void trackingBar_Demo(Mat& image);

	//keyboard response operation image 
	void key_Demo(Mat& image);

	// Self-contained color table operation 
	void colorStyle_Demo(Mat& image);

	//Logical operation of image pixels 
	void bitwise_Demo(Mat& image);

	//Channel separation and merging 
	void channels_Demo(Mat& image);

	//Image color space conversion 
	void inrange_Demo(Mat& image);

	//Image pixel value statistics 
	void pixelStatistic_Demo(Mat& image);

	//Image geometry drawing 
	void drawing_Demo(Mat& image);

	// Randomly draw geometric shapes 
	void randomDrawing_Demo();

	//Polygon filling and drawing 
	void polylineDrawing_Demo();

	//Mouse operation and response 
	void mouseDrawing_Demo(Mat& image);

	//Image pixel type conversion and normalization 
	void norm_Demo(Mat& image);

	//Image scaling and interpolation 
	void resize_Demo(Mat& image);

	//Image flip 
	void flip_Demo(Mat& image);

	//Image rotation 
	void rotate_Demo(Mat& image);

	//Video file camera uses 
	void video_Demo(Mat& image);

	//Video file camera uses 
	void video2_Demo(Mat& image);

	//Video file camera uses RTMP to pull stream 
	void video3_Demo(Mat& image);

	//Image histogram 
	void histogram_Demo(Mat& image);

	//Two-dimensional histogram 
	void histogram2d_Demo(Mat& image);

	//Histogram equalization 
	void histogramEq_Demo(Mat& image);

	//Image convolution operation (blur) 
	void blur_Demo(Mat& image);

	//Gaussian blur 
	void gaussianBlur_Demo(Mat& image);

	//Gauss bilateral blur 
	void bifilter_Demo(Mat& image);

	//real-time face detection 
	void faceDetection_Demo(Mat& image);

	void _array_sum_avx(double* a, double* b, double* re, int ssz);

	void _array_sum(double* a, double* b, double* re, int ssz);

};