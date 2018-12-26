#include "myKinect.h"
#include <iostream>
#include<stack>
//并行计算
#include <windows.h>
#include <ppl.h>

using namespace std;
using namespace cv;
using namespace Concurrency;

int CBodyBasics::frameTimer = 0;
int CBodyBasics::timeIn = 0;
int CBodyBasics::timeOut = 0;
float CBodyBasics::SpineHeightin = 0.0f;
float CBodyBasics::SpineHeightout = 0.0f;
//寻找连通域,筛选并为连通域分类，分为障碍物和人
int CBodyBasics::bwLabel(Mat & src, Mat & dst, vector<Feather> & featherList)
{
	int rows = src.rows;
	int cols = src.cols;

	int labelValue = 0;
	Point seed, neighbor;
	stack<Point> pointStack;    // 堆栈

	int area = 0;               // 用于计算连通域的面积
	int leftBoundary = 0;       // 连通域的左边界，即外接最小矩形的左边框，横坐标值，依此类推
	int rightBoundary = 0;
	int topBoundary = 0;
	int bottomBoundary = 0;
	Rect box;                   // 外接矩形框
	Feather feather;

	featherList.clear();    // 清除数组

	dst.release();
	dst = src.clone();
	for (int i = 0; i < rows; i++)
	{
		uchar *pRow = dst.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (pRow[j] == 255)
			{
				area = 0;
				labelValue++;           // labelValue最大为254，最小为1.
				seed = Point(j, i);     // Point（横坐标，纵坐标）
				dst.at<uchar>(seed) = labelValue;
				pointStack.push(seed);

				area++;
				leftBoundary = seed.x;
				rightBoundary = seed.x;
				topBoundary = seed.y;
				bottomBoundary = seed.y;

				while (!pointStack.empty())
				{
					neighbor = Point(seed.x + 1, seed.y);
					if ((seed.x != (cols - 1)) && (dst.at<uchar>(neighbor) == 255))
					{
						dst.at<uchar>(neighbor) = labelValue;
						pointStack.push(neighbor);

						area++;
						if (rightBoundary < neighbor.x)
							rightBoundary = neighbor.x;
					}

					neighbor = Point(seed.x, seed.y + 1);
					if ((seed.y != (rows - 1)) && (dst.at<uchar>(neighbor) == 255))
					{
						dst.at<uchar>(neighbor) = labelValue;
						pointStack.push(neighbor);

						area++;
						if (bottomBoundary < neighbor.y)
							bottomBoundary = neighbor.y;

					}

					neighbor = Point(seed.x - 1, seed.y);
					if ((seed.x != 0) && (dst.at<uchar>(neighbor) == 255))
					{
						dst.at<uchar>(neighbor) = labelValue;
						pointStack.push(neighbor);

						area++;
						if (leftBoundary > neighbor.x)
							leftBoundary = neighbor.x;
					}

					neighbor = Point(seed.x, seed.y - 1);
					if ((seed.y != 0) && (dst.at<uchar>(neighbor) == 255))
					{
						dst.at<uchar>(neighbor) = labelValue;
						pointStack.push(neighbor);

						area++;
						if (topBoundary > neighbor.y)
							topBoundary = neighbor.y;
					}

					seed = pointStack.top();
					pointStack.pop();
				}
				box = Rect(leftBoundary, topBoundary, rightBoundary - leftBoundary, bottomBoundary - topBoundary);
				rectangle(src, box, 255);
				if( area > threshold_area ){	//排除面积过小的连通域
					feather.area = area;
					feather.boundingbox = box;		
					//cout << body << box.tl() << box.br() << endl;
					//如果矩形框里包含人体一点，则isBody为true
					if (box.contains(body))
						feather.isBody = true;
					else
						feather.isBody = false;
					feather.label = labelValue;
					featherList.push_back(feather);
				}
				
			}
		}
	}
	return labelValue;
}

void CBodyBasics::find_ConnectedDomain(cv::Mat & src_, vector<Feather>& featherList)
{
	//先将Mat转换为IplImage
	IplImage imgTmp = src_;
	IplImage *src = cvCloneImage(&imgTmp);

	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 3);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	//cvThreshold(src, src, 120, 255, CV_THRESH_BINARY);	// 二值化
	// 提取轮廓
	clock_t start, finish;
	start = clock();
	int contour_num = cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(dst);		// 清空数组
	CvSeq *_contour = contour;
	Feather feather;
	int count = 0;
	for (; contour != 0; contour = contour->h_next)
	{

		double tmparea = fabs(cvContourArea(contour));
		if (tmparea < threshold_area)
		{
			cvSeqRemove(contour, 0); // 删除面积小于设定值的轮廓
			continue;
		}
		count++;
		Rect aRect = cvBoundingRect(contour, 0);
		if (aRect.contains(body))
		{
			feather.isBody = true;
		}
		else 
		{
			feather.isBody = false;
		}
		feather.label = count;
		feather.area = tmparea;
		feather.boundingbox = aRect;
		featherList.push_back(feather);
		//if ((aRect.width / aRect.height)<1)
		//{
		//	cvSeqRemove(contour, 0); //删除宽高比例小于设定值的轮廓
		//	continue;
		//}
		/*if (tmparea > maxarea)
		{
			maxarea = tmparea;
		}*/
		rectangle(src_, aRect, 255);
	}
}

//找到每个连通域与摄像机的距离，包括最近和最远的灰度值，用以表示距离,因为我们不需要测得实际距离，我们只需了解连通域的相对距离即可
void CBodyBasics::get_DepthOfConnectedDomain(vector<Feather>& featherList)
{
	parallel_for_each(begin(featherList), end(featherList), [&](Feather &it) {
		it.min_Z = 256;
		it.max_Z = threshold;
		//对boundingbox范围内进行便利，注意行列，x为列，y为行
		for (int i = (it.boundingbox).tl().y; i < (it.boundingbox).br().y; i++) {
			for (int j = (it.boundingbox).tl().x; j < (it.boundingbox).br().x; j++) {
				if ((int)depthImg.at<uchar>(i, j) > it.max_Z && (int)depthImg.at<uchar>(i,j)!=255) {
					it.max_Z = (int)depthImg.at<uchar>(i, j);
				}
				//判断最小值的时候，该点灰度值必须大于二值化的阈值
				if ((int)depthImg.at<uchar>(i, j) > threshold && (int)depthImg.at<uchar>(i, j) < it. min_Z) {
					it.min_Z = (int)depthImg.at<uchar>(i, j);
				}
			}
		}
		//cout << "label" << it.label << endl;
		//cout << "it.max_Z" << it.max_Z << endl;
		//cout << "it.min_Z" << it.min_Z << endl;
	});
	//imshow("depthimg", depthImg);
	//for (vector<Feather>::iterator it = featherList.begin(); it < featherList.end(); it++)
	//{
	//	//借助depthImg获取灰度值
	//	it->min_Z = 256;
	//	it->max_Z = threshold;
	//	//对boundingbox范围内进行便利，注意行列，x为列，y为行
	//	for (int i = (it->boundingbox).tl().y; i < (it->boundingbox).br().y; i++) {
	//		for (int j = (it->boundingbox).tl().x; j < (it->boundingbox).br().x; j++) {
	//			if ((int)depthImg.at<uchar>(i, j) > it->max_Z) {
	//				it->max_Z = (int)depthImg.at<uchar>(i, j);
	//			}
	//			//判断最小值的时候，该点灰度值必须大于二值化的阈值
	//			if ((int)depthImg.at<uchar>(i, j) > threshold && (int)depthImg.at<uchar>(i, j) < it->min_Z) {
	//				it->min_Z = (int)depthImg.at<uchar>(i, j);
	//			}
	//		}
	//	}
	//}
}

//自定义二值化函数
void mask_depth(Mat &image, Mat& th) {
	int nr = image.rows; // number of rows 
	int nc = image.cols; // number of columns 

	for (int i = 0; i<nr; i++) {
		for (int j = 0; j<nc; j++) {
			if ((int)image.at<uchar>(i, j) > threshold) //大于临界值的设为255
			{
				th.at<uchar>(i, j) = (uchar)255;
			}
			else
				th.at<uchar>(i, j) = (uchar)0;
		}
	}
}

//自定义了一个获取深度图障碍物的函数
void find_obstacle(Mat &depth, Mat &res, int thresh = 20, int max_thresh = 255, int area = 100) {

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));  //圆形核，核的大小可适当调整

	//进行开操作，去除小的物体（噪点）
	morphologyEx(depth, depth, MORPH_OPEN,element);
	//blur(depth, out, Size(10, 10));
	//imshow("去除噪点后（depthImg）", depth);

	//得到二值化的图，分理出障碍物
	depth.copyTo(res);  //CopyTo函数，将深度图depth复制到新的dep中去
	mask_depth(depth, res);
}

// 初始化kinect
HRESULT CBodyBasics::InitializeDefaultSensor()
{
	//用于判断每次读取操作的成功与否
	HRESULT hr;

	//搜索kinect
	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr)){
		return hr;
	}

	//找到kinect设备
	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get coordinate mapper and the body reader
		IBodyFrameSource* pBodyFrameSource = NULL;//读取骨架
		IDepthFrameSource* pDepthFrameSource = NULL;//读取深度信息
		IBodyIndexFrameSource* pBodyIndexFrameSource = NULL;//读取背景二值图

		//打开kinect
		hr = m_pKinectSensor->Open();

		//coordinatemapper
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		//bodyframe
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyFrameSource->OpenReader(&m_pBodyFrameReader);
		}

		//depth frame
		if (SUCCEEDED(hr)){
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		}

		//body index frame
		if (SUCCEEDED(hr)){
			hr = m_pKinectSensor->get_BodyIndexFrameSource(&pBodyIndexFrameSource);

		}

		if (SUCCEEDED(hr)){
			hr = pBodyIndexFrameSource->OpenReader(&m_pBodyIndexFrameReader);
			hr = pBodyIndexFrameSource->OpenReader(&m_pBodyIndexFrameReader2);

		}

		SafeRelease(pBodyFrameSource);
		SafeRelease(pDepthFrameSource);
		SafeRelease(pBodyIndexFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		std::cout << "Kinect initialization failed!" << std::endl;
		return E_FAIL;
	}

	//skeletonImg,用于画骨架、背景二值图的MAT
	skeletonImg.create(cDepthHeight, cDepthWidth, CV_8UC3);
	skeletonImg.setTo(0);

	//depthImg,用于画深度信息的MAT
	depthImg.create(cDepthHeight, cDepthWidth, CV_8UC1);
	depthImg.setTo(0);

	return hr;
}

/// Main processing function，将人体分离,得到障碍物
void CBodyBasics::run()
{
	//每次先清空skeletonImg
	skeletonImg.setTo(0);
	//每次先清空barrierImg
	barrierImg.setTo(0);
	//每次先清空depthImg
	depthImg.setTo(0);
	//featherList清空
	featherList.clear();

	//如果丢失了kinect，则不继续操作
	if (!m_pBodyFrameReader)
	{
		return;
	}

	IBodyFrame* pBodyFrame = NULL;//骨架信息
	IDepthFrame* pDepthFrame = NULL;//深度信息
	IColorFrame* pColorFrame = NULL;//彩色信息
	IBodyIndexFrame* pBodyIndexFrame = NULL;//背景二值图

	//记录每次操作的成功与否
	HRESULT hr = S_OK;

	//---------------------------------------获取背景二值图并显示---------------------------------
	if (SUCCEEDED(hr)){
		hr = m_pBodyIndexFrameReader->AcquireLatestFrame(&pBodyIndexFrame);//获得背景二值图信息
	}
	if (SUCCEEDED(hr)){
		BYTE *bodyIndexArray = new BYTE[cDepthHeight * cDepthWidth];//背景二值图是8为uchar，有人是黑色，没人是白色
		pBodyIndexFrame->CopyFrameDataToArray(cDepthHeight * cDepthWidth, bodyIndexArray);

		//把背景二值图画到MAT里
		uchar* skeletonData = (uchar*)skeletonImg.data;
		for (int j = 0; j < cDepthHeight * cDepthWidth; ++j){
			*skeletonData = bodyIndexArray[j]; ++skeletonData;
			*skeletonData = bodyIndexArray[j]; ++skeletonData;
			*skeletonData = bodyIndexArray[j]; ++skeletonData;
		}
		delete[] bodyIndexArray;
	}

	SafeRelease(pBodyIndexFrame);//必须要释放，否则之后无法获得新的frame数据

	//-----------------------获取彩色数据并显示--------------------------
	//int width = 1920;
	//int height = 1080;
	//unsigned int bufferSize = width * height * 4 * sizeof(unsigned char);
	//cv::Mat bufferMat(height, width, CV_8UC4);
	//cv::Mat colorMat(height / 2, width / 2, CV_8UC4);
	//cv::namedWindow("Color");
	//IColorFrameSource* pColorSource; 
	//hr = m_pKinectSensor->get_ColorFrameSource(&pColorSource);
	//  
	//hr = pColorSource->OpenReader(&m_pColorFrameReader);
	//if (SUCCEEDED(hr)) {
	//	hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);
	//	
	//}
	//if (SUCCEEDED(hr)) {
	//	hr = pColorFrame->CopyConvertedFrameDataToArray(bufferSize, reinterpret_cast<BYTE*>(bufferMat.data),ColorImageFormat_Bgra);
	//	if (SUCCEEDED(hr)) {
	//		cv::resize(bufferMat,colorMat,cv::Size(), 0.5,0.5);  
	//	}
	//}
	//SafeRelease(pColorFrame);		//必须要释放，否则之后无法获得新的frame数据
	//cv::imshow("Color", colorMat);

	//-----------------------获取深度数据并显示--------------------------
	//if (SUCCEEDED(hr)){
	//	hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);//获得深度数据
	//}
	//if (SUCCEEDED(hr)){
	//	UINT16 *depthArray = new UINT16[cDepthHeight * cDepthWidth];//深度数据是16位unsigned int
	//	pDepthFrame->CopyFrameDataToArray(cDepthHeight * cDepthWidth, depthArray);
	//	//把深度数据画到MAT中
	//	uchar* depthData = (uchar*)depthImg.data;
	//	for (int j = 0; j < cDepthHeight * cDepthWidth; ++j){
	//		*depthData = depthArray[j];
	//		++depthData;
	//	}
	//	delete[] depthArray;
	//}
	//SafeRelease(pDepthFrame);//必须要释放，否则之后无法获得新的frame数据
	////imshow("depthImg", depthImg);
	//cv::waitKey(5);

	//-----------------------------获取骨架并显示----------------------------
	if (SUCCEEDED(hr)){
		hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);//获取骨架信息	
	}
	if (SUCCEEDED(hr))
	{
		IBody* ppBodies[BODY_COUNT] = { 0 };//每一个IBody可以追踪一个人，总共可以追踪六个人

		if (SUCCEEDED(hr))
		{
			//把kinect追踪到的人的信息，分别存到每一个IBody中
			hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);
		}

		if (SUCCEEDED(hr))
		{
			//对每一个IBody，我们找到他的骨架信息，并且画出来
			ProcessBody(BODY_COUNT, ppBodies);
			//判断人是否跌倒，若跌倒则发出提醒并把isFall设置为true
			detection_isFall(ppBodies, pBodyFrame);
		
		}

		for (int i = 0; i < _countof(ppBodies); ++i)
		{
			SafeRelease(ppBodies[i]);//释放所有
		}
	}
	SafeRelease(pBodyFrame);//必须要释放，否则之后无法获得新的frame数据
	
	//得到障碍物
	get_Barrier(pBodyIndexFrame);
}

//得到障碍物，并框出障碍物和人体
void CBodyBasics::get_Barrier(IBodyIndexFrame* pBodyIndexFrame)
{
	//每次先清空barrierImg
	barrierImg.setTo(0);
	//如果丢失了kinect，则不继续操作
	if (!m_pBodyFrameReader)
	{
		return;
	}

	HRESULT hResult = S_OK;

	//获取source
	IDepthFrameSource* pDepthSource;
	hResult = m_pKinectSensor->get_DepthFrameSource(&pDepthSource);
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::get_DepthFrameSource()" << std::endl;
	}

	// Reader 从source打开reader
	hResult = pDepthSource->OpenReader(&m_pDepthFrameReader);
	if (FAILED(hResult)) {
		std::cerr << "Error : IDepthFrameSource::OpenReader()" << std::endl;
	}
	
	unsigned int bufferSize = cDepthWidth * cDepthHeight * sizeof(unsigned short);
	cv::Mat bufferMat(cDepthHeight, cDepthWidth, CV_16SC1);
	//cv::Mat depthMat(cDepthHeight, cDepthWidth, CV_8UC1);

	//从Reader获取新的frame
	IDepthFrame* pDepthFrame = nullptr;
	hResult = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);
	if (SUCCEEDED(hResult) ) {
		hResult = pDepthFrame->AccessUnderlyingBuffer(&bufferSize, reinterpret_cast<UINT16**>(&bufferMat.data));
		if (SUCCEEDED(hResult)) {
			bufferMat.convertTo(depthImg, CV_8U, -255.0f / 4500.0f, 255.0f);//将缓冲区的转换到depthImg
		}
	}

	//释放该帧，以能够正常接收下一帧
	SafeRelease(pDepthFrame);
	
	//找出障碍物
	find_obstacle(depthImg, barrierImg, 100, 255, 1000);
	//将人移除障碍物的显示图
	//removeBodyFromBarrierImg(pBodyIndexFrame);
	//imshow("去除噪点进行二值化并移除body", barrierImg);

	//-----------------寻找连通区域并将障碍物框出--------------
	Mat dst;
	//cout << "连通域数量： " << bwLabel(barrierImg, dst, featherList) << endl;
	//bwLabel(barrierImg, dst, featherList);
	find_ConnectedDomain(barrierImg, featherList);
	get_DepthOfConnectedDomain(featherList);
	//将"lable"放大，便于观察
	//for (int i = 0; i < dst.rows; i++)
	//{
	//	uchar *p = dst.ptr<uchar>(i);
	//	for (int j = 0; j < dst.cols; j++)
	//	{
	//		p[j] = 30 * p[j];
	//	}
	//}
	////cout << "标号" << "\t" << "面积" << "\t" << "人体" << "\t" << "minZ" << "\t" << "maxZ" << endl;
	//for (vector<Feather>::iterator it = featherList.begin(); it < featherList.end(); it++)
	//{
	//	//cout << it->label << "\t" << it->area << "\t" << it->isBody << "\t" << it->min_Z << "\t" << it->max_Z << endl;
	//	rectangle(dst, it->boundingbox, 255);
	//}
	//imshow("寻找连通域", barrierImg);
	//imshow("找出障碍物和人体", dst);
}

//测量人与障碍物的距离，当距离过小时，发出提醒（未完成，现在默认为只有一个障碍物，当障碍物大于一个时还需分离障碍物）
//横向距离可由像素点的x轴表示，与摄像头的深度距离可由灰度来表示
void CBodyBasics::measure_Distance()
{
	Rect box_ofBody;
	//先找出人体的box和人

	//for (vector<Feather>::iterator it_ = featherList.begin(); it_ < featherList.end(); it_++)
	//{
	//	if (it_->isBody) {
	//		box_ofBody = it_->boundingbox;
	//		//与其他box进行测量
	//		for (vector<Feather>::iterator it = featherList.begin(); it < featherList.end(); it++)
	//		{
	//			if (it->isBody) {
	//				continue;
	//			}
	//	
	//			//横向距离过小并且z方向有交集则发出提醒
	//			if ((abs(it->boundingbox.tl().x - box_ofBody.br().x) <= threshold_Distance || abs(it->boundingbox.br().x - box_ofBody.tl().x) <= threshold_Distance)
	//			&&((it_->max_Z>=it->min_Z&&it_->max_Z<=it->max_Z)||(it_->min_Z>=it->min_Z&&it_->min_Z<=it->max_Z)))
	//			{
	//				Beep(523, 400);//do 
	//			}
	//		}
	//		break;
	//	}
	//}	
	parallel_for_each(begin(featherList), end(featherList), [&](Feather &it_){
		if (it_.isBody) {
			box_ofBody = it_.boundingbox;
			parallel_for_each(begin(featherList), end(featherList), [&](Feather &it) {
				if (it.isBody) {
					return;
				}
				//横向距离过小并且z方向有交集则发出提醒
				if ((abs(it.boundingbox.tl().x - box_ofBody.br().x) <= threshold_Distance || abs(it.boundingbox.br().x - box_ofBody.tl().x) <= threshold_Distance)
					&& ((it_.max_Z >= it.min_Z&&it_.max_Z <= it.max_Z) || (it_.min_Z >= it.min_Z&&it_.min_Z <= it.max_Z)))
				{
					//向左走或是向右走
					Beep(523, 400);//do 

					if (it.boundingbox.height <= box_ofBody.height / 3&&it.boundingbox.height<=threshold_Width)
					{
						//可以跨越

					}

				}
			});
			return;
		}
			
	});

}

//得到跌倒信息
bool CBodyBasics::get_IsFall()
{
	return isFall;
}

// 检测是否摔倒 通过检测中心点的下降速度
void CBodyBasics::detection_isFall(IBody** ppBodies,IBodyFrame *pBodyFrame)
{
	HRESULT hr;
	if (get_Joints(ppBodies) != NULL)
	{
		//cout << frameTimer << endl;
		//cout << "LeftFoot:" << get_Joints(ppBodies)[JointType_FootRight].Position.Y << endl;
		Joint *joints;//存储关节点类
		joints = get_Joints(ppBodies);
		//得到初始时间
		if (frameTimer == 0)
		{
			timeIn = GetTickCount();
			SpineHeightin = joints[JointType_SpineMid].Position.Y;
		}
		frameTimer++;
		
		if (frameTimer == frame_Update)
		{
			timeOut = GetTickCount();
			SpineHeightout = joints[JointType_SpineMid].Position.Y;
			float  SpineV = 1000 * (SpineHeightin - SpineHeightout) / (timeOut - timeIn);
			/*cout << "SpineHeightin:" << SpineHeightin << endl;
			cout << "SpineHeightout:" << SpineHeightout << endl;
			cout <<"SpineV:" <<SpineV << endl;*/
			//system("pause");
			if (SpineV > threshold_Speed)
			{
			////	求得的地面方程常数项不能正确获得，故计算两髋中心点距离右脚掌的高度baseH来代替
			//	Vector4 floorClipPlane;
			//	hr = pBodyFrame->get_FloorClipPlane(&floorClipPlane);
			////	计算人此时的高度
			//	float baseH;
			//	baseH = fabsf(floorClipPlane.x*joints[JointType_SpineBase].Position.X + floorClipPlane.y*joints[JointType_SpineBase].Position.Y
			//		+ floorClipPlane.z*joints[JointType_SpineBase].Position.Z+joints[JointType_FootRight].Position.Y) / sqrtf(pow(floorClipPlane.x, 2) + pow(floorClipPlane.y, 2)
			//			+ pow(floorClipPlane.z, 2));
				float baseH=0.0f;
				//baseH = joints[JointType_SpineMid].Position.Y - joints[JointType_FootRight].Position.Y;
				baseH = joints[JointType_SpineMid].Position.Y;
				/*cout << "高度" << baseH << endl;
				cout << "joints[JointType_SpineMid].Position.Y:" << joints[JointType_SpineMid].Position.Y << endl;
				cout << "joints[JointType_FootRight].Position.Y:" << joints[JointType_FootRight].Position.Y << endl;
				system("pause");*/
				if (baseH < threshold_Height) {
					set_IsFall(true);		//将摔倒信息置为true
					//Beep(523, 400);//do 
					/*cout << "身体中心向下的速度是：   " << SpineV << "m/s" << endl;
					cout << "高度" << baseH << endl;
					system("pause");*/
				}		

			}
			//有关数据都归零
			frameTimer = 0;
			timeIn = 0;
			timeOut = 0;
			SpineHeightin = 0.0f;
			SpineHeightout = 0.0f;
		}
		if (joints[JointType_Head].Position.Y<joints[JointType_HandRight].Position.Y) {
			set_IsFall(false);
			//cout << joints[JointType_Head].Position.Y << " " << joints[JointType_HandRight].Position.Y << endl;
		}

	}
	else 
	{
		//cout << "没得到" << endl;
		frameTimer = 0;
	}
}

//更改跌倒信息
void CBodyBasics::set_IsFall(bool res)
{
	isFall = res;
}
//得到人的骨骼点
Joint * CBodyBasics::get_Joints(IBody ** ppBodies)
{
	
	for (int i = 0; i < BODY_COUNT; ++i)
	{
		IBody *pBody=ppBodies[i];
		HRESULT hr;	//记录操作是否成功
		if (pBody)
		{
			BOOLEAN bTracked = false;
			hr = pBody->get_IsTracked(&bTracked);	//判断人是否被追踪

			if (SUCCEEDED(hr) && bTracked)
			{
				Joint joints[JointType_Count];//存储关节点类
				
				//获得关节点类
				hr = pBody->GetJoints(_countof(joints), joints);
				//遇到被追踪的即返回
				if (SUCCEEDED(hr))
				{
					return joints;
				}
			}
		}
	}
	return NULL;
}

/// Handle new body data
void CBodyBasics::ProcessBody(int nBodyCount, IBody** ppBodies)
{
	//记录操作结果是否成功
	HRESULT hr;

	//对于每一个IBody
	for (int i = 0; i < nBodyCount; ++i)
	{
		IBody* pBody = ppBodies[i];
		if (pBody)//还没有搞明白这里pBody和下面的bTracked有什么区别
		{
			BOOLEAN bTracked = false;
			hr = pBody->get_IsTracked(&bTracked);

			if (SUCCEEDED(hr) && bTracked)
			{
				Joint joints[JointType_Count];//存储关节点类
				HandState leftHandState = HandState_Unknown;//左手状态
				HandState rightHandState = HandState_Unknown;//右手状态

				//获取左右手状态
				pBody->get_HandLeftState(&leftHandState);
				pBody->get_HandRightState(&rightHandState);

				//存储深度坐标系中的关节点位置
				DepthSpacePoint *depthSpacePosition = new DepthSpacePoint[_countof(joints)];

				//获得关节点类
				hr = pBody->GetJoints(_countof(joints), joints);
				if (SUCCEEDED(hr))
				{
					for (int j = 0; j < _countof(joints); ++j)
					{
						//将关节点坐标从摄像机坐标系（-1~1）转到深度坐标系（424*512）
						m_pCoordinateMapper->MapCameraPointToDepthSpace(joints[j].Position, &depthSpacePosition[j]);
					}
					//输出距离
					//cout << "摄像头距离:" << joints[0].Position.Y << endl;
				//	cout << "横向距离" << depthSpacePosition[0].X << endl;
					//取得人的一点
					body = Point(depthSpacePosition[0].X, depthSpacePosition[0].Y);
					
					//------------------------hand state left-------------------------------
					DrawHandState(depthSpacePosition[JointType_HandLeft], leftHandState);
					DrawHandState(depthSpacePosition[JointType_HandRight], rightHandState);

					//---------------------------body-------------------------------
					DrawBone(joints, depthSpacePosition, JointType_Head, JointType_Neck);
					DrawBone(joints, depthSpacePosition, JointType_Neck, JointType_SpineShoulder);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_SpineMid);
					DrawBone(joints, depthSpacePosition, JointType_SpineMid, JointType_SpineBase);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_ShoulderRight);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_ShoulderLeft);
					DrawBone(joints, depthSpacePosition, JointType_SpineBase, JointType_HipRight);
					DrawBone(joints, depthSpacePosition, JointType_SpineBase, JointType_HipLeft);

					// -----------------------Right Arm ------------------------------------ 
					DrawBone(joints, depthSpacePosition, JointType_ShoulderRight, JointType_ElbowRight);
					DrawBone(joints, depthSpacePosition, JointType_ElbowRight, JointType_WristRight);
					DrawBone(joints, depthSpacePosition, JointType_WristRight, JointType_HandRight);
					DrawBone(joints, depthSpacePosition, JointType_HandRight, JointType_HandTipRight);
					DrawBone(joints, depthSpacePosition, JointType_WristRight, JointType_ThumbRight);

					//----------------------------------- Left Arm--------------------------
					DrawBone(joints, depthSpacePosition, JointType_ShoulderLeft, JointType_ElbowLeft);
					DrawBone(joints, depthSpacePosition, JointType_ElbowLeft, JointType_WristLeft);
					DrawBone(joints, depthSpacePosition, JointType_WristLeft, JointType_HandLeft);
					DrawBone(joints, depthSpacePosition, JointType_HandLeft, JointType_HandTipLeft);
					DrawBone(joints, depthSpacePosition, JointType_WristLeft, JointType_ThumbLeft);

					// ----------------------------------Right Leg--------------------------------
					DrawBone(joints, depthSpacePosition, JointType_HipRight, JointType_KneeRight);
					DrawBone(joints, depthSpacePosition, JointType_KneeRight, JointType_AnkleRight);
					DrawBone(joints, depthSpacePosition, JointType_AnkleRight, JointType_FootRight);

					// -----------------------------------Left Leg---------------------------------
					DrawBone(joints, depthSpacePosition, JointType_HipLeft, JointType_KneeLeft);
					DrawBone(joints, depthSpacePosition, JointType_KneeLeft, JointType_AnkleLeft);
					DrawBone(joints, depthSpacePosition, JointType_AnkleLeft, JointType_FootLeft);
				}
				delete[] depthSpacePosition;
			}
		}
	}
	//cv::imshow("skeletonImg", skeletonImg);
	//cv::waitKey(5);
}

//画手的状态
void CBodyBasics::DrawHandState(const DepthSpacePoint depthSpacePosition, HandState handState)
{
	//给不同的手势分配不同颜色
	CvScalar color;
	switch (handState){
	case HandState_Open:
		color = cvScalar(255, 0, 0);
		break;
	case HandState_Closed:
		color = cvScalar(0, 255, 0);
		break;
	case HandState_Lasso:
		color = cvScalar(0, 0, 255);
		break;
	default://如果没有确定的手势，就不要画
		return;
	}

	circle(skeletonImg,
		cvPoint(depthSpacePosition.X, depthSpacePosition.Y),
		20, color, -1);
}

// Draws one bone of a body (joint to joint)
void CBodyBasics::DrawBone(const Joint* pJoints, const DepthSpacePoint* depthSpacePosition, JointType joint0, JointType joint1)
{
	TrackingState joint0State = pJoints[joint0].TrackingState;
	TrackingState joint1State = pJoints[joint1].TrackingState;

	// If we can't find either of these joints, exit
	if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked))
	{
		return;
	}

	// Don't draw if both points are inferred
	if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred))
	{
		return;
	}

	CvPoint p1 = cvPoint(depthSpacePosition[joint0].X, depthSpacePosition[joint0].Y),
		p2 = cvPoint(depthSpacePosition[joint1].X, depthSpacePosition[joint1].Y);

	// We assume all drawn bones are inferred unless BOTH joints are tracked
	if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked))
	{
		//非常确定的骨架，用白色直线
	    line(skeletonImg, p1, p2, cvScalar(255, 255, 255));
	}
	else
	{
		//不确定的骨架，用红色直线
		line(skeletonImg, p1, p2, cvScalar(0, 0, 255));
	}
}

/// Constructor
CBodyBasics::CBodyBasics() :
m_pKinectSensor(NULL),
m_pCoordinateMapper(NULL),
m_pBodyFrameReader(NULL){}

/// Destructor
CBodyBasics::~CBodyBasics()
{
	SafeRelease(m_pBodyFrameReader);
	SafeRelease(m_pCoordinateMapper);

	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}
	SafeRelease(m_pKinectSensor);
}

//在显示障碍物的mat中移出人（用不到）
//void CBodyBasics::removeBodyFromBarrierImg(IBodyIndexFrame* pBodyIndexFrame)
//{
//	BYTE *bodyIndexArray = new BYTE[cDepthHeight * cDepthWidth];//背景二值图是8位uchar，有人是黑色，没人是白色
//	pBodyIndexFrame->CopyFrameDataToArray(cDepthHeight * cDepthWidth, bodyIndexArray);
//
//	//把背景二值图与障碍物的图作比较，将人所在位置的灰度变为0；
//	/*cout << bodyIndexArray[150] << endl;*/
//	for (int i = 0; i < cDepthHeight; ++i) {
//		for (int j = 0; j < cDepthWidth; ++j) {
//			if ((int)bodyIndexArray[i*cDepthWidth + j] != 255) {
//				barrierImg.at<uchar>(i, j) = (uchar)0;
//			}
//			/*if ((int)skeletonImg.at<uchar>(i, j) == 0) {
//			barrierImg.at<uchar>(i, j) = (uchar)0;
//			}*/
//		}
//	}
//	delete[] bodyIndexArray;
//}
