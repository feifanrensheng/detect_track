#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

const int CONTOUR_MAX_AERA = 3;

int main ( int argc, char** argv )
{
	//声明IplImage指针
	IplImage* pFrame = NULL;
	IplImage* pFrImg = NULL;
	IplImage* pBkImg = NULL;
	CvMat* pFrameMat = NULL;
	CvMat* pFrMat = NULL;
	CvMat* pBkMat = NULL;

	CvCapture* pCapture = NULL;

	int nFrmNum = 0;

	//创建窗口 
	cvNamedWindow("video", 1);
	cvNamedWindow("background", 1);
	cvNamedWindow("foreground", 1);

	//使窗口有序排列 
	cvMoveWindow("video", 30, 0);
	cvMoveWindow("background", 360, 0);
	cvMoveWindow("foreground", 690, 0);
	//argc = 1;

	if ( argc > 2 )
	{
		fprintf(stderr, "Usage: bkgrd [video_file_name]\n" );
		return -1;
	}

	//打开摄像头 
	if (argc == 1)
	{
		pCapture = cvCaptureFromCAM(0);
		if( !pCapture )
		{
			fprintf(stderr, "Can not open camera.\n" );
			return -2;
		}
	}
	//打开视频文件
	
	if (argc == 2)
	{
		pCapture = cvCaptureFromFile(argv[1]);
		if( !(pCapture = cvCaptureFromFile(argv[1])))
		{
			fprintf(stderr, "Can not open video file %s\n", argv[1] );
			return -2;
		}		
	}


	//逐帧读取视频      
  	while(pFrame = cvQueryFrame( pCapture ))      
    {      
    	nFrmNum++;  
    	//如果是第一帧，需要申请内存，并初始化
    	if(nFrmNum == 1)
    	{
    		pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U,1); 
    		pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U,1); 
    		pBkMat    = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1); 
    		pFrMat    = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);  
    		pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
    		//转化成单通道图像再处理 
    		cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
    		cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY); 
    		cvConvert(pFrImg, pFrameMat); 
    		cvConvert(pFrImg, pFrMat); 
    		cvConvert(pFrImg, pBkMat); 
    	}
    	else
    	{
    		cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY); //颜色空间转换
    		cvConvert(pFrImg, pFrameMat);            //用于图像和矩阵之间的相互转换
    		//先高斯滤波，以平滑图像 
    		cvSmooth(pFrameMat, pFrameMat, CV_GAUSSIAN, 3, 0, 0);  

    		//当前帧跟背景图相减 
    		cvAbsDiff(pFrameMat, pBkMat, pFrMat); //计算两个数组差的绝对值的函数
    		//二值化前景图 
    		cvThreshold(pFrMat, pFrImg, 60, 255.0, CV_THRESH_BINARY); //该函数的典型应用是对灰度图像进行阈值操作得到二值图像
    		//进行形态学滤波，去掉噪音   
    		cvErode(pFrImg, pFrImg, 0, 1); 
    		cvDilate(pFrImg, pFrImg, 0, 1); 
    		//更新背景  
    		cvRunningAvg(pFrameMat, pBkMat, 0.003, 0); 
    		//将背景转化为图像格式，用以显示  
    		cvConvert(pBkMat, pBkImg);

        // 下面的程序段用来找到轮廓
        CvMemStorage *stor;
        CvSeq *cont, *result, *squares;
        CvSeqReader reader;
        

        stor = cvCreateMemStorage(0);
        cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), stor);

        // 找到所有轮廓
        cvFindContours( pFrImg, stor, &cont, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

        for (;cont;cont = cont->h_next)
        {
          CvRect r = ((CvContour*)cont)->rect;
          if (r.height * r.width > CONTOUR_MAX_AERA) //面积小的方形抛弃掉
          {
            cvRectangle( pFrame, cvPoint(r.x,r.y), cvPoint(r.x + r.width, r.y + r.height), CV_RGB(255,0,0), 1, CV_AA, 0);
          }

        }
        cvReleaseMemStorage( &stor );
       
    		//显示图像  
    		cvShowImage("video", pFrame); 
    		cvShowImage("background", pBkImg);      
    		cvShowImage("foreground", pFrImg);   
    		//如果有按键事件，则跳出循环 
    		//此等待也为cvShowImage函数提供时间完成显示 
    		//等待时间可以根据CPU速度调整    
    		//if ( cvWaitKey(0) ) continue;
    		if (cvWaitKey(33) >= 0) break;

   		}
   }
   //销毁窗口   
   cvDestroyWindow("video"); 
   cvDestroyWindow("background");
   cvDestroyWindow("foreground");  
   //释放图像和矩阵
   cvReleaseImage(&pFrImg); 
   cvReleaseImage(&pBkImg);
   cvReleaseMat(&pFrameMat);
   cvReleaseMat(&pFrMat); 
   cvReleaseMat(&pBkMat);
   cvReleaseCapture(&pCapture);  
   return 0;
}