#include <iostream>
#include<vector>
#include<thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<windows.h>
#include "myKinect.h"
#include <mmsystem.h>


//boost::thread
//#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )	//不显示控制台
using namespace std;
using namespace cv;

//将三张图整合到一个页面
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size = cvSize(512, 424));
//将单通道转换为三通道
Mat convertTo3Channels(const Mat& binImg);

int a = 0;

int main() 
{
	CBodyBasics myKinect;
	HRESULT hr = myKinect.InitializeDefaultSensor();
	//PlaySound(TEXT("bgm"), NULL, SND_FILENAME | SND_ASYNC);
	if (SUCCEEDED(hr)) {
		while (1) {
			//auto startTime = std::chrono::high_resolution_clock::now();
			//剥离出人体,障碍物
			/*thread t(&CBodyBasics::f, ref(myKinect));
			t.join();*/
			myKinect.run();
			//显示窗口	
			vector<Mat> imgs(3);
			imgs[0] = convertTo3Channels(myKinect.depthImg);
			imgs[1] = convertTo3Channels(myKinect.barrierImg);
			imgs[2] = myKinect.skeletonImg;
			MultiImage_OneWin("Multiple Images", imgs, cvSize(2, 2));//排版为2*2

			 //测量人与障碍物的距离
			myKinect.measure_Distance();

			
			if (myKinect.get_IsFall())
			{
				Beep(523, 400);
				//cout << "跌倒了" << endl;
				//PlaySound(TEXT("bgm"), NULL, SND_FILENAME | SND_ASYNC);
			}
			if (cv::waitKey(30) == VK_ESCAPE) {
				break;
			}

			//auto endTime = std::chrono::high_resolution_clock::now();
			//float totalTime = std::chrono::duration<float, std::milli>
			//	(endTime - startTime).count();

			//cout <<"运行时间："<< totalTime << endl;

		}
	}
	else {
		cout << "kinect initialization failed!" << endl;
		system("pause");
	}
	
	return 0;
}


Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
	vector<Mat> channels;
	//将原图存三次在存入三通道图即可
	for (int i = 0; i<3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}
void MultiImage_OneWin(const string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size)
{
	//显示窗口
	Mat Disp_Img;
	
	//原图像的宽高
	CvSize Img_OrigSize = cvSize(SrcImg_V[0].cols, SrcImg_V[0].rows);
	//------------------ 算出窗口的width和height------------------------
	//Width vs height ratio of source image
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(50, 50);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//--------------------空白处设置--- ----------------------------
	CvSize DispBlank_Edge = cvSize(0, 0);
	CvSize DispBlank_Gap = cvSize(5, 5);
	//-----------------------得到window的大小-------------------------------
	Disp_Img.create(Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_8UC3);
	Disp_Img = Scalar(128, 128, 128);//设置背景颜色
	//Left top position for each image
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);//左上角的位置
	CvPoint LT_Pos = LT_BasePos;

	//显示所有Mat
	int Img_Num = (int)SrcImg_V.size();
	for (int i = 0; i < Img_Num; i++)
	{
		//Obtain the left top position
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		//Writting each to Window's Image
		Mat imgROI = Disp_Img(Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		resize(SrcImg_V[i], imgROI, Size(ImgDisp_Size.width, ImgDisp_Size.height));

		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}

	//得到屏幕的宽高
	int Scree_W = GetSystemMetrics(SM_CXSCREEN);
	int Scree_H = GetSystemMetrics(SM_CYSCREEN);
	
	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//窗口在屏幕中央
	
	//--------------绘制文字部分------------
	//写出文字的部分
	char text[100];
	itoa(a, text, 10);
	a = a + 1;
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 1;
	int thickness = 1;
	int baseline;
	//获取文本框的长宽
	cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

	//绘制
	cv::Point origin;
	origin.x = Disp_Img.cols / 1.5 - text_size.width / 2;
	origin.y = Disp_Img.rows / 1.5 - text_size.height / 2;
	cv::putText(Disp_Img, text, origin, font_face, font_scale,255, thickness, 8, 0);
	//----------------文字部分结束----------------

	imshow(MultiShow_WinName.c_str(), Disp_Img);
	
}




