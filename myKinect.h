#pragma once
#include <Kinect.h>
#include <opencv2\opencv.hpp>
#include<vector>

//播放音频
#include<dsound.h>
#include <chrono>
#pragma comment(lib, "WINMM.LIB")

#define threshold 130	//自定义二值化函数的阈值
#define threshold_area 5000	//大于该值则判断该连通域有效，否则判断为噪点
#define threshold_Distance 50//单位为分辨率
#define frame_Update 5	//每frame_Update,就重新计算是否跌倒时的速度,即有关变量都变为0
#define threshold_Speed 0.6	//跌倒时速度的阈值
#define threshold_Height 0.2 //跌倒时高度的阈值
#define threshold_Width	100	//可以跨越时的宽度阈值



typedef struct _Feather
{
	int label;					// 连通域的label值
	int area;					// 连通域的面积
	cv::Rect boundingbox;       // 连通域的外接矩形框
	bool isBody;			//判断连通域的种类，如果是障碍物则为false，如果为人体，则为true
	int min_Z;
	int max_Z;				//设摄像头方向为z轴，用灰度值表示相对距离
} Feather;
// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

class CBodyBasics
{
	//kinect 2.0 的深度空间的高*宽是 424 * 512，在官网上有说明
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;

public:
	CBodyBasics();
	~CBodyBasics();
	void                    run();//获得骨架、背景二值图和深度信息
	void                    get_Barrier(IBodyIndexFrame*);//识别并剥离障碍物
	void					measure_Distance();//测量人与障碍物的距离，当距离过小时，发出提醒
	bool                    get_IsFall();	//得到是否跌倒的信息
	HRESULT                 InitializeDefaultSensor();//用于初始化kinect
	static void             f(CBodyBasics &a) { a.run(); }
	static void             f2(CBodyBasics &a) { a.measure_Distance(); }
	
   //显示图像的Mat
	cv::Mat skeletonImg;	//人体图像
	cv::Mat depthImg;		//深度图像
	cv::Mat barrierImg;		//障碍物图像
	cv::Mat colorImg;		//彩色图像
	//存储人的一点，用来判断连通域的种类，即人或障碍物
	cv::Point body;
	// 存放连通域特征
	vector<Feather> featherList;

private:
	IKinectSensor*          m_pKinectSensor;//kinect源
	ICoordinateMapper*      m_pCoordinateMapper;//用于坐标变换
	IBodyFrameReader*       m_pBodyFrameReader;//用于骨架数据读取
	IDepthFrameReader*      m_pDepthFrameReader;//用于深度数据读取
	IColorFrameReader*      m_pColorFrameReader;//用于彩色数据读取
	IBodyIndexFrameReader*  m_pBodyIndexFrameReader;//用于背景二值图读取
	IBodyIndexFrameReader*  m_pBodyIndexFrameReader2;//用于背景二值图读取
	static int frameTimer;							//用于定位帧数
	static int timeIn;								//用于计算速度的起始时间
	static int timeOut;								//用于计算速度的末尾时间
	static float SpineHeightin;
	static float SpineHeightout;						//分别存储初始和末尾的高度
	bool isFall=false;

	//通过获得到的信息，把骨架和背景二值图画出来
	void ProcessBody(int nBodyCount, IBody** ppBodies);
	//画骨架函数
	void DrawBone(const Joint* pJoints, const DepthSpacePoint* depthSpacePosition, JointType joint0, JointType joint1);
	//画手的状态函数
	void DrawHandState(const DepthSpacePoint depthSpacePosition, HandState handState);
	//寻找连通域
	int bwLabel(cv::Mat & src,cv:: Mat & dst, vector<Feather> & featherList);
	//寻找连通域改良版
	void find_ConnectedDomain(cv::Mat &src, vector<Feather> &featherList);
	//找到每个连通域与摄像机的距离，包括最近和最远的灰度值，用以表示距离
	void get_DepthOfConnectedDomain(vector<Feather> & featherList);
	//检测是否摔倒
	void detection_isFall(IBody** ppBodies,IBodyFrame *pBodyFrame);
	//将摔倒置为true
	void set_IsFall(bool res);
	//得到人的骨骼点
	Joint *get_Joints(IBody** ppBodies);		
	//bool relieve_Fall(IBody** ppBodies, IBodyFrame *pBodyFrame);
	//在显示障碍物的mat中移出人
	//void removeBodyFromBarrierImg(IBodyIndexFrame*);
};
