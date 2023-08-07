# include  <iostream>
# include <stdio.h> # include <ctime> # include <sys/timeb.h> # include <opencv2/opencv.hpp> # include <opencv2/features2d.hpp> # include <opencv2/imgproc /imgproc.hpp> # include <assert.h> # include <xmmintrin.h> # include <immintrin.h> # include <intrin.h> # include "clock_.h" # include "QuickDemo.h" 













using  namespace cv;
using namespace std;

#define MAX_PATH_LEN 256

void  mat_example() {
	Mat src1_gray = imread("E:\\project_file\\1.jpg", 0); //Read the image and convert it to single channel
	Mat src1_color = imread("E:\\project_file\\1.jpg"); / /Read in the image and convert it to 3 channels
		//imshow("src1_gray", src1_gray); //imshow("src1_color", src1_color);	


		Mat src1_gray_1 = src1_gray; //shallow copy
	Mat src1_gray_2;
	src1_gray.copyTo(src1_gray_2); // deep copy

	//Single-channel image pixel traversal, set the photo paper value in the upper left quarter to 0 
//for (int i = 0; i < src1_gray.rows / 2; i++) //{ // for (int j = 0; j < src1_gray.cols / 2; j++) // { // src1_gray.at<uchar>(i, j) = 0;//Use at to operate on single channel i row j column pixels // } / /} //imshow("src1_gray after pixel operation", src1_gray); //imshow("src1_gray_1", src1_gray_1);//Since it is a shallow copy and shares the same memory data, the operation on src1_gray affects src1_gray_1; // imshow("src1_gray_2", src1_gray_2);//Since it is a deep copy, src1_gray_2 is a real copy with one copy of data, so it is not affected by the image of src1_gray;	










	//Image addition and subtraction operations, multiplication and division operations are similar, you can write your own code to view the effect 
//Mat mat_add = src1_gray + src1_gray_2; //imshow("Image addition", mat_add); //Mat mat_sub = src1_gray_2 - src1_gray; / /imshow("Image subtraction", mat_sub);	




	//Single-channel image is converted to three-channel, other conversions, such as three-channel to single-channel operation are similar, you can check it online
	Mat mat_3chanel;
	cvtColor(src1_gray, src1_gray, COLOR_GRAY2RGB);

	imshow("Single-channel image converted to three-channel", src1_gray);

	//Data type conversion, uchar conversion to float type, other conversions, such as uchar conversion to double, float conversion to uchar can be checked online;
	Mat mat_float;
	src1_gray.convertTo(mat_float, CV_32F);

	//Image ROI operation (slicing) 
//Rect rec(0, 0, 300, 500); //Mat mat_color_roi = src1_color(rec); //imshow("Image ROI operation", mat_color_roi);	




	///Image mask operation;
	Mat mask;
	threshold(src1_gray_2, mask, 128, 255, 0);
	imshow("mask", mask);
	Mat src_color_mask;
	src1_color.copyTo(src_color_mask, mask);
	imshow("mask operation", src_color_mask);

	waitKey();
}

void  mat_my_test() {
	Mat src = imread("E:\\project_file\\1.jpg"); //read the image and convert it to 3 channels
	Mat my, mycpy, cpy;
	src.copyTo(cpy); src.copyTo(my); src.copyTo(mycpy);
	auto p1 = my.data;
	auto p2 = mycpy.data;

	struct  timeb timep;
	ftime(&timep);
	auto now = timep.time * 1000 + timep.millitm, last = now;

	src += cpy; //The essence is cv::add, cuda parallel computing may be called in core.hpp 
//imshow("opencv+=", src);	

	ftime(&timep); now = timep.time * 1000 + timep.millitm;
	printf("opencv cp cost : %d ms\n", now - last); last = now;




	for (int i = 0; i < my.rows; i++)
	{
		for (int j = 0; j < my.cols; j++)
		{
			my.at <Vec3b>(i, j) += mycpy.at <Vec3b>(i, j);
		}
	}
	//imshow("my +=", my); 
//waitKey(); ftime (&timep); now = timep.time * 1000 + timep.millitm;	


	printf("my at function cpy cost : %d ms\n", now - last); last = now;
	/*
		   opencv cp cost : 1 ms
		   my at function cpy cost : 79 ms
	   */ //The read image is converted to 1 channel 	src = imread ( "E:\\project_file\\1.jpg" , 0 );	


	src.copyTo(cpy); src.copyTo(my); src.copyTo(mycpy);

	now = timep.time * 1000 + timep.millitm, last = now;

	src += cpy;

	ftime(&timep); now = timep.time * 1000 + timep.millitm;
	printf("opencv 1 channel cp cost : %d ms\n", now - last); last = now;

	for (int i = 0; i < my.rows; i++)
	{
		for (int j = 0; j < my.cols; j++)
		{
			my.at <uchar>(i, j) = mycpy.at <uchar>(i, j); //Use at to operate on single channel i row j column pixels
		}
	}

	ftime(&timep); now = timep.time * 1000 + timep.millitm;
	printf("my at 1 channel cp cost : %d ms\n", now - last); last = now;

	/*
		opencv 1 channel cp cost : 28 ms
		my at 1 channel cp cost : 85 ms
	*/

}

void  quikdemo_test() {
	QuickDemo ep;
	Mat img(imread("D:\\project_file\\i1.jpg"));
	Mat img2(imread("D:\\project_file\\i2.jpg"));
	ep.Sift(img, img2);
}

int  main() {

	return  0;

}
QuickDemo {
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