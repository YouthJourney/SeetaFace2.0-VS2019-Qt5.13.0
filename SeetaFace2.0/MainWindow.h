#pragma once

#include <QtWidgets/QMainWindow>
#include "x64/Debug/uic/ui_MainWindow.h"
#include "ui_MainWindow.h"
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QImage>
#include <QDebug>
#include <QMessageBox>

#include <thread>
#include <vector>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include "SeetaFace2/FaceDetector/include/seeta/FaceDetector.h"
#include "SeetaFace2/FaceLandmarker/include/seeta/FaceLandmarker.h"
#include "SeetaFace2/FaceRecognizer/include/seeta/FaceDatabase.h"
#include "SeetaFace2/FaceDetector/include/seeta/CFaceInfo.h"

#include "Struct_cv.h"

#pragma execution_character_set("utf-8")

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);
	~MainWindow();

public slots:
	//导入图片1
	void importFace1();
	//导入图片2
	void importFace2();
	//共有的图片导入接口
	void importFace(int flag);
	//获取人脸相似对
	void startSeetaFace();
	//显示框选出的人脸，显示5点、81点特征点
	void startFaceAlignment();

private:
	Ui::MainWindowClass ui;

	QImage* face1;
	QImage* face2;
	cv::Mat face1Mat;
	cv::Mat face2Mat;

	seeta::FaceDatabase* FDB;
	std::vector<SeetaPointF> face1Points;
	std::vector<SeetaPointF> face2Points;
	bool faceFlag;//两图中是否有人脸

	float result;//比对结果
	std::mutex myMutex;

private:
	//QImage转Mat
	cv::Mat QImage2cvMat(const QImage& inImage, bool inCloneImageData = true);
	//Mat转QImage
	QImage cvMat2QImage(const cv::Mat& inMat);
	//特征抽取，并赋值给feat
	void faceDetectionFunc(cv::Mat& mat, std::vector<SeetaPointF>& facePoints, std::string windName);
	//特征比较
	float getSimilar(cv::Mat& face1, std::vector<SeetaPointF>& face1Points, cv::Mat& face2, std::vector<SeetaPointF>& face2Points);
};
