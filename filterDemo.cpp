#include "filterDemo.h"
//QuickDemo.cpp

//# include  "QuickDemo.h"

void  QuickDemo::Sift(Mat& imageL, Mat& imageR) {
	//Method of extracting feature points //SIFT 	cv::Ptr<cv::SIFT> sift = cv::SIFT:: create ();
	//cv::Ptr <cv::SIFT> sift = cv::SIFT::Creat(); //OpenCV 4.4.0 and later versions //ORB //cv::Ptr<cv::ORB> sift = cv::ORB:: create(); //SURF //cv::Ptr<cv::SURF> surf = cv::features2d::SURF::create();	







	   //Feature points
	std::vector<cv::KeyPoint> keyPointL, keyPointR;
	//Extract feature points separately
	sift->detect(imageL, keyPointL);
	sift->detect(imageR, keyPointR);

	////Draw feature points 
//cv::Mat keyPointImageL; //cv::Mat keyPointImageR; //drawKeypoints(imageL, keyPointL, keyPointImageL, cv::Scalar::all(-1), cv::DrawMatchesFlags: :DRAW_RICH_KEYPOINTS); //drawKeypoints(imageR, keyPointR, keyPointImageR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);	




	////Display feature points 
//cv::imshow("KeyPoints of imageL", keyPointImageL); //cv::imshow("KeyPoints of imageR", keyPointImageR);	


	// feature point matching
	cv::Mat despL, despR;
	//Extract feature points and calculate feature descriptor
	sift->detectAndCompute(imageL, cv::Mat(), keyPointL, despL);
	sift->detectAndCompute(imageR, cv::Mat(), keyPointR, despR);

	//Struct for DMatch: query descriptor index, train descriptor index, train image index and distance between descriptors. 
//int queryIdx –> is the subscript of the feature point descriptor (descriptor) of the test image, and is also the feature point corresponding to the descriptor (keypoint) subscript. //int trainIdx –> is the subscript of the feature point descriptor of the sample image, and also the subscript of the corresponding feature point. //int imgIdx –> Useful when the sample is multiple images. //float distance –> represents the Euclidean distance between the pair of matching feature point descriptors (essentially a vector). The smaller the value, the more similar the two feature points are.	



	std::vector<cv::DMatch> matches;

	//If the flannBased method is used, the type of desp obtained through the orb is different, and the type needs to be converted first 
	if (despL.type() != CV_32F || despR.type() != CV_32F)
	{
		despL.convertTo(despL, CV_32F);
		despR.convertTo(despR, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->match(despL, despR, matches);

	// Calculate the maximum value of feature point distance 
	double maxDist = 0;
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
			maxDist = dist;
	}

	//Select a good matching point
	std::vector< cv::DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++) {
		if (matches[i].distance < 0.5 * maxDist) {
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat imageOutput;
	cv::drawMatches(imageL, keyPointL, imageR, keyPointR, good_matches, imageOutput);

	cv::namedWindow("picture of matching");
	cv::imshow("picture of matching", imageOutput);
	//imwrite("D:\\project_file\\i3.jpg", imageOutput);	

}

void  QuickDemo::Hue(Mat& src) {

	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("src", src);
	Mat dsth;

	int dims = 1;

	float hranges[] = { 0 , 255 };
	const float* ranges[] = { hranges };
	int size = 256, channels = 0;

	// Calculate the histogram 
	calcHist(&src, 1, &channels, Mat(), dsth,
		dims, &size, ranges);

	int scale = 1;

	Mat dst(size * scale, size, CV_8U, Scalar(0));

	double mv = 0, mx = 0;

	minMaxLoc(dsth, &mv, &mx, 0, 0);

	//drawing 
	int hpt = saturate_cast <int>(0.9 * size);
	for (int i = 0; i < 256; ++i) {
		float binvalue = dsth.at < float >(i);

		int realvalue = saturate_cast <int>(binvalue * hpt / mx);
		rectangle(dst, Point(i * scale, size - 1),
			Point((i + 1) * scale - 1, size - realvalue),
			Scalar(rand() & 255, rand() & 255, rand() & 255));
	}

	imshow("Hue", ​​dst);

}

void  QuickDemo::Hue_Saruration(Mat& src) {
	imshow("Original image", src);
	Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	int huebinnum = 30, saturationbinnum = 32;
	int hs_size[] = { huebinnum, saturationbinnum };

	//Define the hue range 
	float hueRanges[] = { 0 , 180 };
	//Define the saturation range from 0 (black, white, gray) to 255 (pure spectrum color float saRanges[] = { 0 , 256 } ;
	const float* ranges[] = { hueRanges, saRanges };

	MatND dst;

	int channel[] = { 0 , 1 };

	calcHist(&hsv, 1, channel, Mat(), dst, 2, hs_size, ranges, true, false);

	double mv = 0;
	minMaxLoc(dst, 0, &mv, 0, 0);
	int scale = 10;

	Mat hs_img = Mat::zeros(saturationbinnum * scale, huebinnum * 10, CV_8UC3);

	for (int i = 0; i < huebinnum; ++i)
		for (int j = 0; j < saturationbinnum; ++j) {
			float binv = dst.at < float >(i, j);

			int intensity = cvRound(binv * 255 / mv); //intensity

			rectangle(hs_img, Point(i * scale, j * scale),
				Point((i + 1) * scale - 1, (j + 1) * scale),
				Scalar::all(intensity), FILLED);

		}
	imshow("HS histogram", hs_img);



}

/*
Mat g_mask, g_src;Point prevPt(-1, -1);
static void on_Mouse(int event, int x, int y, int flag, void *) {
	if (x < 0 || x >= g_src.cols || y < 0 || y >= g_src.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flag & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flag & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(g_mask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(g_src, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow("----程序窗口----", g_src);
	}

};

void QuickDemo::water_shed(Mat &src) {
	//cvtColor(src, src, COLOR_BGR2GRAY);


	imshow("----程序窗口----", src);

	src.copyTo(g_src);

	Mat gray;
	cvtColor(g_src, g_mask, COLOR_BGR2GRAY);
	cvtColor(g_mask, gray, COLOR_GRAY2BGR);

	g_mask = Scalar::all(0);

	setMouseCallback("----程序窗口----", on_Mouse, 0);

	while (true) {
		int c = waitKeyEx(0);

		if ((char)c == 27) break;

		if ((char)c == '2') {
			g_mask = Scalar::all(0);
			src.copyTo(g_src);
			imshow("image", g_src);
		}
			//若检测到按键值为 1 或者空格，则进行处理
		if ((char)c == '1' || (char)c == ' ') {
			//定义一些参数
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			//寻找轮廓
			findContours(g_mask, contours, hierarchy,
				RETR_CCOMP, CHAIN_APPROX_SIMPLE);

			//轮靡为空时的处理
			if (contours.empty())
				continue;
			//复制掩膜
			Mat maskImage(g_mask.size(), CV_32S);
			maskImage = Scalar::all(0);
			//循环绘制出轮廓
			for (int index = 0; index >= 0; index = hierarchy[index][0], compCount++)

				drawContours(maskImage, contours, index, Scalar::all(compCount + 1),
					-1, 8, hierarchy, INT_MAX);

			//compCount 为零时的处理if( compCount == 0 )
			continue;
			//生成随机颜色
			vector<Vec3b> colorTab;
			for (i = 0; i < compCount; i++) {
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);

				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}

			//计算处理时间并输出到窗口中
			double dTime = (double)getTickCount();
			watershed(src, maskImage);
			dTime = (double)getTickCount() - dTime;
			printf("\t处理时间 = ggms\n", dTime*1000. / getTickFrequency());
			//双层循环，将分水岭图像遍历存入 watershedImage 中
			Mat watershedImage(maskImage.size(), CV_8UC3);

			for (i = 0; i < maskImage.rows; i++)
				for (j = 0; j < maskImage.cols; j++) {
					int index = maskImage.at<int>(i, j);
					if (index == -1)
						watershedImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
					else if (index <= 0 || index > compCount)watershedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0); else
					watershedImage.at
						<Vec3b>(i, j) = colorTab[index - 1];
				}
			/ / Mix grayscale image and watershed effect image and display the final window
			watershedImage = watershedImage * 0.5 + gray * 0.5;
			imshow("watershed transform", watershedImage);
		}
	}
 }
 */





void  QuickDemo::find_contours(Mat& src) {

	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("src", src);

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

	src = src > 118;
	imshow("Thresholding 118", src);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(src, contours, hierarchy, RETR_CCOMP,
		CHAIN_APPROX_SIMPLE);

	int id = 0;

	while (id >= 0) {
		printf("%d\n", (int)contours[id].size());
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, id, color, FILLED, 8,
			hierarchy);
		id = hierarchy[id][0];
	}

	imshow("find_contours", dst);

}

void  QuickDemo::equa_hist(Mat& src) {

	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("src", src);
	Mat dst;

	equalizeHist(src, dst);

	imshow("equalizeHist", dst);

}

void  QuickDemo::SHT(Mat& src) {
	imshow("src", src);

	Mat mid, dst, tmp;

	Canny(src, mid, 50, 200, 3);
	cvtColor(mid, dst, COLOR_GRAY2BGR);

	// make the transformation

	tmp.create(Size(mid.cols, mid.rows), 1);

	imshow("canny", mid);

	vector<Vec2f> lines;
	HoughLines(mid, lines, 1, CV_PI / 180, 50, 0, 0);



	// draw line segment

	cout << lines.size() << '\n';
	for (size_t i = 0; i < lines.size(); ++i) {
		float rho = lines[i][0], theta = lines[i][1];
		Point p1, p2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;

		p1.x = cvRound(x0 + 1000 * (-b));
		p1.y = cvRound(y0 + 1000 * (a));
		p2.x = cvRound(x0 - 1000 * (-b));
		p2.y = cvRound(y0 - 1000 * (a));

		line(dst, p1, p2, Scalar(55, 100, 195), 1, LINE_AA);
	}

	imshow("Hough", dst);

}

void  QuickDemo::scharr_edge(Mat& src) {
	imshow("src", src);

	Mat grad_x, grad_y, abs_x, abs_y;

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// X direction 
	Scharr(src, grad_x, CV_16S, 1, 0, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_x);
	imshow("x direction", abs_x);

	//y direction 
	Scharr(src, grad_y, CV_16S, 0, 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_y);
	imshow("y direction", abs_y);

	//merge gradient
	Mat dst;
	addWeighted(abs_x, 0.5, abs_y, 0.5, 0, dst);
	imshow("merged", dst);

}

void  QuickDemo::laplacian_edge(Mat& src) {
	imshow("src", src);

	Mat gray, dst, abs_dst;

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(src, gray, COLOR_RGB2GRAY);

	Laplacian(gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(dst, abs_dst);

	imshow("laplacian_edge", abs_dst);

}

void  QuickDemo::sobel_edge(Mat& src) {
	imshow("src", src);

	Mat grad_x, grad_y, abs_x, abs_y;



	// Sobel in x direction (src, grad_x, CV_16S, 1 , 0 , 3 , 1 , 1 , BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_x);
	imshow("x direction", abs_x);

	//y direction 
	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_y);
	imshow("y direction", abs_y);

	//merge gradient
	Mat dst;
	addWeighted(abs_x, 0.5, abs_y, 0.5, 0, dst);
	imshow("merged", dst);

}

void  QuickDemo::canny_edge_plus(Mat& src) {
	imshow("src", src);
	Mat dst, edge, gray;

	dst.create(src.size(), src.type());

	cvtColor(src, gray, COLOR_BGR2GRAY);

	blur(gray, edge, Size(3, 3));

	Canny(edge, edge, 3, 9, 3);

	dst = Scalar::all(0);
	src.copyTo(dst, edge);

	imshow("canny_edge_plus", dst);
}

void  QuickDemo::canny_edge(Mat& src) {
	imshow("src", src);
	Mat up;

	Canny(src, up, 150, 100, 3);

	imshow("canny_edge", up);
}

void  QuickDemo::pyr_down(Mat& img) {
	imshow("src", img);
	Mat up;

	pyrDown(img, up, Size(img.cols / 2, img.rows / 2));

	imshow("pyr_up", up);
}

void  QuickDemo::pyr_up(Mat& img) {
	imshow("src", img);
	Mat up;

	pyrUp(img, up, Size(img.cols * 2, img.rows * 2));

	imshow("pyr_up", up);
}

void  QuickDemo::flood_fill(Mat& img) {
	Rect comp;
	imshow("src", img);
	floodFill(img, Point(50, 50), Scalar(155, 10, 55),
		&ccomp, Scalar(5, 5, 5), Scalar(5, 5, 5));
	imshow("flood_fill", img);
}

/*
 - op
	- MORPH_OPEN
	- MORPH_CLOSE
	- MORPH_GRADIENT
	- MORPH_TOPHAT -
	MORPH_BLACKHAT
	- MORPH_ERODE
	- MORPH_DILATE
 */
void QuickDemo::morpho(Mat& img, int op) {
	Mat out;
	imshow("src", img);
	morphologyEx(img, out, op,
		getStructuringElement(MORPH_RECT, Size(5, 5)));
	imshow("out", out);
}

void  QuickDemo::morpho_gradient(Mat& img) {
	imshow("src", img);
	Mat di, er;
	dilate(img, di, getStructuringElement(MORPH_RECT, Size(5, 5)));
	erode(img, er, getStructuringElement(MORPH_RECT, Size(5, 5)));
	di -= er;
	imshow("morpho_gradient", di);
}

void  QuickDemo::top_hat(Mat& img) {
	imshow("src", img);

	Mat out, top_hat;
	img.copyTo(top_hat);
	erode(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	dilate(out, img, getStructuringElement(MORPH_RECT, Size(5, 5)));
	top_hat -= img;
	imshow("top_hat", top_hat);
}

void  QuickDemo::black_hat(Mat& img) {
	imshow("src", img);

	Mat out, tmp;
	img.copyTo(tmp);
	dilate(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	erode(out, img, getStructuringElement(MORPH_RECT, Size(5, 5)));
	img -= tmp;
	imshow("black_hat", img);
}

void  QuickDemo::op_opening() {
	Mat img(imread("E:\\project_file\\b.jpg"));
	//ep.file_storage(); imshow ( "before" , img);		

	Mat out;

	erode(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	dilate(out, img, getStructuringElement(MORPH_RECT, Size(5, 5)));
	imshow("after", img);

}

void  QuickDemo::op_closing() {
	Mat img(imread("E:\\project_file\\b.jpg"));
	//ep.file_storage(); imshow ( "before" , img);		

	Mat out;

	dilate(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	erode(out, img, getStructuringElement(MORPH_RECT, Size(5, 5)));

	imshow("after", img);

}

void  QuickDemo::img_dilate() {
	Mat img(imread("E:\\project_file\\b.jpg"));
	//ep.file_storage(); imshow ( "before" , img);		

	Mat out;

	dilate(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	imshow("after", out);
}

void  QuickDemo::img_erode() {
	Mat img(imread("E:\\project_file\\b.jpg"));
	//ep.file_storage(); imshow ( "before" , img);		

	Mat out;

	erode(img, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	imshow("after", out);
}

void  QuickDemo::file_storage() {
	FileStorage fs("E:\\project_file\\s.txt", FileStorage::WRITE);

	Mat R = Mat_<uchar >::eye(3, 3);
	Mat T = Mat_< double >::zeros(3, 1);

	fs << "R" << R;
	fs << "T" << T;
	fs.release();
}

//ROI_AddImage 
void QuickDemo::ROI_AddImage() {
	Mat logeImg = imread("E:\\project_file\\b.jpg");
	Mat img = imread("E:\\project_file\\1.jpg");

	Mat ImgRoi = img(Rect(0, 0, logeImg.cols, logeImg.rows));

	Mat mask(imread("E:\\project_file\\b.jpg", 0));

	imshow("<1>ImgRoi", ImgRoi);

	logImg.copyTo(ImgRoi, mask);

	imshow("<2>ROI achieves image overlay instance window", img);

}

QuickDemo::~QuickDemo() {
	waitKey();
}

//Space color conversion Demo 
void QuickDemo::colorSpace_Demo(Mat& image) {

	Mat hsv, gray;

	cvtColor(image, hsv, COLOR_RGB2HSV);
	cvtColor(image, gray, COLOR_RGB2GRAY);
	imshow("HSV", hsv); //Hue hue, Saturation saturation, Value is hue imshow ( "grayscale" , gray);
	/ /imwrite("F:/OpenCV/Image/hsv.png", hsv); //imwrite("F:\\OpenCV\\Image\\gray.png", gray);		


}

//Mat creates image 
void QuickDemo::matCreation_Demo(Mat& image) {
	//Mat m1, m2; //m1 = image.clone(); //image.copyTo(m2); //imshow("Image 1", m1); //imshow("Image 2", m2); 






	Mat m3 = Mat::zeros(Size(400, 400), CV_8UC3);
	m3 = Scalar(255, 255, 0);
	cout << "width:" << m3.cols << "height:" << m3.rows << " channels:" << m3.channels() << endl;
	//cout << m3 << endl; imshow ( "image3" , m3);	


	Mat m4 = m3;
	//imshow("image 4", m4);

	m4 = Scalar(0, 255, 255);
	imshow("Image33", m3);

	//imshow("image 44", m4);
}

//Image pixel reading and writing 
void QuickDemo::pixelVisit_Demo(Mat& image) {
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();


	//for (int row = 0; row < h; row++) {
	//	for (int col = 0; col < w; col++) {
	//		if (dims == 1) { //灰度图像
	//			int pv = image.at<uchar>(Point(row, col));
	//			image.at<uchar>(Point(row, col)) = 255 - saturate_cast<uchar>(pv);
	//		}
	//		else if (dims == 3) { //彩色图像
	//			Vec3b bgr = image.at<Vec3b>(row, col);
	//			image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
	//			image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
	//			image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
	//		}
	//	}
	//}

	uchar* img_prt = image.ptr <uchar>();
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			for (int dim = 0; dim < dims; dim++) {
				*img_prt++ = 255 - *img_prt;
			}
			//if (dims == 1) { // grayscale image 
// *img_prt++ = 255 - *img_prt; //} //else if (dims == 3) { // color image // *img_prt++ = 255 - *img_prt; // *img_prt++ = 255 - *img_prt; // *img_prt++ = 255 - *img_prt;			





		}
	}
	imshow("image pixel reading and writing demo", image);
}


//Image pixel arithmetic operation 
void QuickDemo::operators_Demo(Mat& image) {

	Mat dst;
	Mat m = Mat::zeros(image.size(), image.type());

	m = Scalar(50, 50, 50);
	add(image, m, dst);
	imshow("Add operation", dst);

	m = Scalar(50, 50, 50);
	subtract(image, m, dst);
	imshow("Subtract operation", dst);

	m = Scalar(2, 2, 2);
	multiply(image, m, dst);
	imshow("multiplication operation", dst);

	m = Scalar(2, 2, 2);
	divide(image, m, dst);
	imshow("division operation", dst);

}


//Scroll bar callback function 
void onTrack(int b, void* userdata) {

	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());

	if (b > 100) {
		m = Scalar(b - 100, b - 100, b - 100);
		add(image, m, dst);
	}
	else {
		m = Scalar(100 - b, 100 - b, 100 - b);
		subtract(image, m, dst);
	}

	//addWeighted(image, 1.0, m, 0, b, dst);

	imshow("Brightness and contrast adjustment", dst);
}

//Scroll bar callback function 
void onContrast(int b, void* userdata) {

	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());

	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);

	imshow("Brightness and contrast adjustment", dst);
}

//Scroll bar to adjust image brightness 
void QuickDemo::trackingBar_Demo(Mat& image) {
	int max_value = 200;
	int lightness = 100;
	int contrast_value = 100;


	namedWindow("Brightness and contrast adjustment", WINDOW_AUTOSIZE);

	createTrackbar("Value Bar", "Brightness and Contrast Adjustment", &lightness, max_value, onTrack, (void*)&image);
	createTrackbar("Contrast Bar", "Brightness and Contrast Adjustment", &contrast_value, max_value, onContrast, (void*)&image);

	onTrack(lightness, &image);
}


//keyboard response operation image 
void QuickDemo::key_Demo(Mat& image) {

	Mat dst = Mat::zeros(image.size(), image.type());
	while (true) {
		int c = waitKey(100);
		if (c == 27) { //ESC exit break ;					

		}
		else  if (c == 49) {
			cout << "key #1" << endl;
			cvtColor(image, dst, COLOR_RGB2GRAY);
		}
		else  if (c == 50) {
			cout << "key #2" << endl;
			cvtColor(image, dst, COLOR_RGB2HSV);
		}
		else  if (c == 51) {
			cout << "key #3" << endl;
			dst = Scalar(50, 50, 50);
			add(image, dst, dst);
		}
		imshow("keyboard response", dst);
	}
}

// Self-contained color table operation 
void QuickDemo::colorStyle_Demo(Mat& image) {

	Mat dst = Mat::zeros(image.size(), image.type());
	int index = 0;
	int pixNum = 0;
	while (true) {
		int c = waitKey(2000);
		if (c == 27) {
			break;
		}
		else  if (c == 49) {
			String pixPath = "./Image/color";
			pixPath = pixPath.append(to_string(pixNum++));
			pixPath = pixPath.append(".png");
			imwrite(pixPath, dst);
		}
		applyColorMap(image, dst, (index++) % 19);
		imshow("color style", dst);
	}
}

//Logical operation of image pixels 
void QuickDemo::bitwise_Demo(Mat& image) {

	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	bitwise_and(m1, m2, dst);
	imshow("pixel bit-and operation", dst);
	bitwise_or(m1, m2, dst);
	imshow("pixel bit-or operation", dst);
	bitwise_xor(m1, m2, dst);
	imshow("Pixel XOR operation", dst);
}

//Channel separation and merging 
void QuickDemo::channels_Demo(Mat& image) {

	vector<Mat> mv;
	split(image, mv);
	//imshow("blue", mv[0]); //imshow("green", mv[1]); //imshow("red", mv[2]);	



	Mat dst;
	vector<Mat> mv2;

	//mv[1] = 0; 
//mv[2] = 0; //merge(mv, dst); //imshow("blue", dst);	



	mv[0] = 0;
	mv[2] = 0;
	merge(mv, dst);
	imshow("green", dst);

	//mv[0] = 0; 
//mv[1] = 0; //merge(mv, dst); //imshow("red", dst);	



	int from_to[] = { 0 , 2 , 1 , 1 , 2 , 0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("Channel Mix", dst);

}

// Image color space conversion 
void QuickDemo::inrange_Demo(Mat& image) {

	Mat hsv;
	cvtColor(image, hsv, COLOR_RGB2HSV);
	imshow("hsv", hsv);
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
	//imshow("mask", mask); bitwise_not (mask, mask);
	imshow("mask", mask);


	Mat readback = Mat::zeros(image.size(), image.type());
	readback = Scalar(40, 40, 200);
	image.copyTo(readback, mask);
	imshow("roi region extraction", readback);
}

//Image pixel value statistics 
void QuickDemo::pixelStatistic_Demo(Mat& image) {
	double minv, maxv;

	Point minLoc, maxLoc;
	vector<Mat> mv;
	split(image, mv);
	for (int i = 0; i < mv.size(); i++) {
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc);
		cout << "No." << i << "min:" << minv << "max:" << maxv << endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);
	cout << "means:" << mean << endl;
	cout << "stddev:" << stddev << endl;
}

//Image geometry drawing 
void QuickDemo::drawing_Demo(Mat& image) {

	Mat bg = Mat::zeros(image.size(), image.type());

	rectangle(bg, Rect(250, 100, 100, 150), Scalar(0, 0, 255), -1, 8, 0);
	circle(bg, Point(300, 175), 50, Scalar(255, 0, 0), 1, 8, 0);
	line(bg, Point(250, 100), Point(350, 250), Scalar(0, 255, 0), 4, 8, 0);
	line(bg, Point(350, 100), Point(250, 250), Scalar(0, 255, 0), 4, 8, 0);
	ellipse(bg, RotatedRect(Point2f(200.0, 200.0), Size2f(100.0, 200.0), 00.0), Scalar(0, 255, 255), 2, 8);
	imshow("bg", bg);

	Mat dst;
	addWeighted(image, 1.0, bg, 0.3, 0, dst);
	imshow("Geometry Drawing", dst);
}

// Randomly draw geometric shapes 
void QuickDemo::randomDrawing_Demo() {

	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	RNG rng(12345);
	int w = canvas.cols;
	int h = canvas.rows;

	while (true) {
		int c = waitKey(10);
		if (c == 27) {
			break;
		}

		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		//canvas = Scalar(0, 0, 0); 
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 1, 8, 0);
		imshow(" Random drawing demo", canvas);
	}
}

//Polygon filling and drawing 
void QuickDemo::polylineDrawing_Demo() {

	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 280);
	Point p4(320, 450);
	Point p5(80, 400);
	vector<Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);

	Point p11(50, 50);
	Point p12(200, 50);
	Point p13(250, 150);
	Point p14(160, 300);
	Point p15(30, 350);
	vector<Point> pts1;
	pts1.push_back(p11);
	pts1.push_back(p12);
	pts1.push_back(p13);
	pts1.push_back(p14);
	pts1.push_back(p15);

	//fillPoly(canvas, pts, Scalar(255, 255, 0), 8, 0); 
//polylines(canvas, pts, true, Scalar(0, 255, 255), 2, LINE_AA, 0);	

	vector<vector<Point>> contours;
	contours.push_back(pts);
	contours.push_back(pts1);
	drawContours(canvas, contours, -1, Scalar(255, 0, 0), 4);

	imshow("polygon", canvas);
}


static  void  onDraw(int event, int x, int y, int flags, void* userdata)
{
	static Point sp(-1, -1), ep(-1, -1);
	Mat srcImg = *((Mat*)userdata);
	Mat image = srcImg.clone();

	if (event == EVENT_LBUTTONDOWN) {
		sp.x = x;
		sp.y = y;
		//cout << "down_x = " << sp.x << " dwon_y = " << sp.y << endl;
	}
	else  if (event == EVENT_LBUTTONUP) {
		ep.x = x;
		ep.y = y;
		if (ep.x > image.cols) {
			ep.x = image.cols;
		}
		if (ep.y > image.rows) {
			ep.y = image.rows;
		}
		//cout << "up_x = " << ep.x << " up_y = " << ep.y << endl; 
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
			Rect box(sp.x, sp.y, dx, dy);
			//rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0 ); //imshow("Mouse drawing", image); imshow ( "ROI area" , image (box));												


			sp.x = -1;
			sp.y = -1;
		}

	}
	else  if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			//cout << "up_x = " << ep.x << " up_y = " << ep.y << endl;

			if (ep.x > image.cols) {
				ep.x = image.cols;
			}
			if (ep.y > image.rows) {
				ep.y = image.rows;
			}
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				//srcImg.copyTo(image); 				image = srcImg.clone ();
				Rect box(sp.x, sp.y, dx, dy);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("mouse draw", image);

			}


		}
	}
}

//Mouse operation and response 
void QuickDemo::mouseDrawing_Demo(Mat& image) {
	namedWindow("mouse drawing", WINDOW_AUTOSIZE);
	setMouseCallback("mouse drawing", onDraw, (void*)(&image));
	imshow("mouse drawing", image);

}

//Image pixel type conversion and normalization 
void QuickDemo::norm_Demo(Mat& image) {

	Mat dst;
	cout << image.type() << endl;
	//CV_8UC3 is converted to CV_32FC3 	image.convertTo (image, CV_32F );	

	cout << image.type() << endl;
	normalize(image, dst, 1.0, 0.0, NORM_MINMAX);
	cout << dst.type() << endl;
	imshow(" Image pixel normalization", dst);
}

//Image scaling and interpolation 
void QuickDemo::resize_Demo(Mat& image) {

	Mat zoomSmall, zoomLarge;
	int w = image.cols;
	int h = image.rows;

	resize(image, zoomSmall, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	imshow("zoomSmall", zoomSmall);

	resize(image, zoomLarge, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
	imshow("zoomLarge", zoomLarge);
}

//Image flip 
void QuickDemo::flip_Demo(Mat& image) {

	Mat dst;
	flip(image, dst, 0); //flip up and down (reflection in water) 
//flip(image, dst, 1);//flip left and right (mirror image) //flip(image, dst, -1);// 180° flip (diagonal flip) imshow ( "image flip" , dst);	


}

//Image rotation 
void QuickDemo::rotate_Demo(Mat& image) {

	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = getRotationMatrix2D(Point(w / 2, h / 2), 60, 2);
	double cos = abs(M.at < double >(0, 0));
	double sin = abs(M.at < double >(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at < double >(0, 2) += nw / 2 - w / 2;
	M.at < double >(1, 2) += nh / 2 - h / 2;
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));
	imshow("image rotation", dst);
}

//Video file camera use 
void QuickDemo::video_Demo(Mat& image) {
	VideoCapture capture(0);
	//VideoCapture capture("./Image/sample.mp4"); 

	Mat frame;

	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			cout << "frame empty" << endl;
			break;
		}
		flip(frame, frame, 1); //Flip the video image left and right 
		imshow("camera real-time monitoring", frame);

		// TODO: do something ... 
//mouseDrawing_Demo(frame);//Video image screenshot //colorSpace_Demo(frame);//HSV GRAY		


		int c = waitKey(10);
		if (c == 27) {
			break;
		}

	}

	//release Release the camera resource 
	capture.release();
}

//Video file camera use 
void QuickDemo::video2_Demo(Mat& image) {
	//VideoCapture capture(0); VideoCapture capture ( "./Image/lane.avi" ) ;
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH); // Video width int frame_height = capture.get (CAP_PROP_FRAME_HEIGHT); //Video height int count = capture.get (CAP_PROP_FRAME_COUNT); // Total video frames double fps = capture.get (CAP_PROP_FPS); //FPS Refresh frames per second double fourcc = capture.get ( CAP_PROP_FOURCC); 





	   //Video encoding format
	cout << "frame width:" << frame_width << endl;
	cout << "frame height:" << frame_height << endl;
	cout << "frames sum:" << count << endl;
	cout << "FPS:" << fps << endl;
	cout << "frame fourcc:" << fourcc << endl;
	VideoWriter writer("./video/lane_save.avi", fourcc, fps, Size(frame_width, frame_height), true);

	Mat frame;

	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			cout << "frame empty" << endl;
			break;
		}
		//flip(frame, frame, 1);//Flip the video image left and right 
		imshow("camera real-time monitoring", frame);
		writer.write(frame);

		// TODO: do something ... 
//mouseDrawing_Demo(frame);//Video image screenshot //colorSpace_Demo(frame);//HSV GRAY		


		int c = waitKey(10);
		if (c == 27) {
			break;
		}

	}

	//release Release the camera resource 
	capture.release();
	writer.release();
}

//The video file camera uses RTMP to pull the stream 
void QuickDemo::video3_Demo(Mat& image) {
	//VideoCapture capture(0); 

	VideoCapture vcap;
	Mat frame;

	string videoStreamAddress = "rtmp://192.168.254.104:1935/live/live"; if
		(!vcap.open(videoStreamAddress)) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	while (true) {
		vcap.read(frame);
		if (frame.empty()) {
			cout << "frame empty" << endl;
			break;
		}
		flip(frame, frame, 1); //Flip the video image left and right 
		imshow("RTMP", frame);

		int c = waitKey(10);
		if (c == 27) {
			break;
		}

	}
	//release Release the camera resource 
	vcap.release();
}

//Image histogram 
void QuickDemo::histogram_Demo(Mat& image) {
	//Three-channel separation 

	vector<Mat> bgr_plane;
	split(image, bgr_plane);

	//Define parameter variables 
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0 , 255 };
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;

	// Calculate the histogram of the Blue, Green, and Red channels 
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	//Define the histogram window 
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(Size(hist_w, hist_h), CV_8UC3);

	//Normalized histogram data 
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//Draw a histogram curve 
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at < float >(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at < float >(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at < float >(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at < float >(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at < float >(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at < float >(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	//Display the histogram 
	namedWindow("Histogrma Demo", WINDOW_AUTOSIZE);
	imshow("Histogrma Demo", histImage);
}

//Two-dimensional histogram 
void QuickDemo::histogram2d_Demo(Mat& image) {
	//2D histogram 

	Mat hsv, hs_hist;
	cvtColor(image, hsv, COLOR_RGB2HSV); //Convert to HSV image 
	int hbins = 30, sbins = 32; //Division ratio 180/30=5 256/32=8 int hist_bins[] = { hbins,sbins };
	float h_range[] = { 0 , 180 }; //H range float s_range[] = { 0 , 256 }; //S range const float * hs_ranges[] = { h_range,s_range }; //range pointer pointer int hs_channels [] = { 0 , 1 }; //Channel number calcHist (&hsv, 1 , hs_channels,	




	Mat(), hs_hist, 2, hist_bins, hs_ranges); //Two-dimensional histogram conversion double 
	maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
	int scale = 10;
	//Define the 2D histogram window 	Mat hist2d_image = Mat:: zeros (sbins*scale, hbins*scale, CV_8UC3);
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++) {
			float binVal = hs_hist.at < float _
			>(h, s);
			//Amount of proportion int intensity = cvRound (binVal * 255 / maxVal);
			//Draw a rectangle rectangle (hist2d_image, Point (h*scale, s*scale),
			Point((h + 1) * scale - 1, (s + 1) * scale - 1),


				Scalar::all(intensity),
				-1 );
		}
	}
	//Image color conversion 
	applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
	imshow("HS Histogram", hist2d_image);
	imwrite("./Image/hist_2d.png", hist2d_image);
}

//Histogram equalization 
void QuickDemo::histogramEq_Demo(Mat& image) {

	Mat gray;
	cvtColor(image, gray, COLOR_RGB2GRAY);
	imshow("grayscale image", gray);
	Mat dst;
	equalizeHist(gray, dst);
	imshow("Histogram Equalization Demonstration", dst);

	//Color image histogram equalization 
//Mat hsv; //cvtColor(image, hsv, COLOR_RGB2HSV); //vector<Mat> hsvVec; //split(hsv, hsvVec); //equalizeHist(hsvVec[2], hsvVec[2]); //Mat hsvTmp; //merge(hsvVec, hsvTmp); //Mat dst; //cvtColor(hsvTmp, dst, COLOR_HSV2RGB); //imshow("Histogram equalization demo", dst) ;	









}

//Image convolution operation (blur) 
void QuickDemo::blur_Demo(Mat& image) {

	Mat dst;
	blur(image, dst, Size(13, 13), Point(-1, -1));
	imshow("image blur", dst);
}
//Gaussian Blur 
void QuickDemo::gaussianBlur_Demo(Mat& image) {

	Mat dst;
	GaussianBlur(image, dst, Size(7, 7), 15);
	imshow("Gaussian Blur", dst);
}

//Gauss bilateral blur 
void QuickDemo::bifilter_Demo(Mat& image) {

	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	imshow("Gaussian bilateral blur", dst);
	imwrite("C:\\Users\\fujiahuang\\Desktop\\b.jpg", dst);
}
/*
实时人脸检测
void QuickDemo::faceDetection_Demo(Mat &image)
{
	string root_dir = "D://x86//opencv4.6//opencv//sources//samples//dnn//face_detector//";
	Net net = readNetFromTensorflow(root_dir + "opencv_face_detector_uint8.pb", root_dir + "opencv_face_detector.pbtxt");
	VideoCapture capture(0);
	//VideoCapture capture;
	//string videoStreamAddress = "rtmp://192.168.254.104:1935/live/live";
	//if (!capture.open(videoStreamAddress)) {
	//	cout << "Error opening video stream or file" << endl;
	//	return;
	//}

	Mat frame;

	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			cout << "frame empty" << endl;
			break;
		}
		flip(frame, frame, 1);//视频图像左右翻转
		Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
		net.setInput(blob);//NCHW
		Mat probs = net.forward();
		Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

		//解析结果
		int num = 0;
		float confidence = 0.0;
		float fTemp = 0.0;
		for (int i = 0; i < detectionMat.rows; i++) {
			confidence = detectionMat.at<float>(i, 2);
			if (confidence > 0.5) {
				fTemp = confidence;
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.cols);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.cols);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);

				num++;
			}
		}
		//Mat dst;
		//bilateralFilter(frame, dst, 0, 100, 10);//高斯双边模糊

		putText(frame, "NO." + to_string(num) + " SSIM:" + to_string(fTemp), Point(30, 50), FONT_HERSHEY_TRIPLEX, 1.3, Scalar(26, 28, 124), 4);

		imshow("人脸实时检测", frame);
		int c = waitKey(1);
		if (c == 27) {
			break;
		}
	}
}

人脸照片检测
void QuickDemo::faceDetection_Demo(Mat &image)
{
	string root_dir = "D:/opencv4.5.0/opencv/sources/samples/dnn/face_detector/";
	Net net = readNetFromTensorflow(root_dir + "opencv_face_detector_uint8.pb", root_dir + "opencv_face_detector.pbtxt");

	Mat frame;
	frame = image.clone();

	while (true) {
		frame = image.clone();
		//flip(frame, frame, 1);//视频图像左右翻转
		Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
		net.setInput(blob);//NCHW
		Mat probs = net.forward();
		Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

		//解析结果
		int num = 0;

		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > 0.5) {
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.cols);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.cols);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);

				num++;
			}
		}

		putText(frame, "NO." + to_string(num) + " pcs", Point(30, 50), FONT_HERSHEY_TRIPLEX, 1.3, Scalar(124, 28, 26), 2);

		imshow("人脸实时检测", frame);

		int c = waitKey(1000);
		if (c == 27) {
			break;
		}
	}
 }+*/


void QuickDemo::_array_sum_avx(double* a, double* b, double* re, int ssz)
{
	clock_t;
	__m256d m1, m2;

	//for (int k = 0; k < 4; ++ k) 
//{ for ( int i = 0 ; i < ssz; i += 4 )	

	{
		m1 = _mm256_set_pd(a[Cv16suf::i], a[i + 1], a[i + 2], a[i + 3]);
		m2 = _mm256_set_pd(b[i], b[i + 1], b[i + 2], b[i + 3]);

		__m256d l1 = _mm256_mul_pd(m1, m2);

		re[i + 3] = l1.m256d_f64[0];
		re[i + 2] = l1.m256d_f64[1];
		re[i + 1] = l1.m256d_f64[2];
		re[i] = l1.m256d_f64[3];
	}

	size_t en = clock();
	t.show("avx");
}

void QuickDemo::_array_sum(double* a, double* b, double* re, int ssz)
{
	clock_t;
	//for (int k = 0; k < 4; ++ k) 
//{ for ( int i = 0 ; i < ssz; ++i)	

	{
		re[i] = a[i] * b[i];
	}

	t.show("normal cpu cost");
}
