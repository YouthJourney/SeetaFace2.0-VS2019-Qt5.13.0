#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	connect(ui.importFace1Button, &QPushButton::clicked, this, &MainWindow::importFace1);
	connect(ui.importFace2Button, &QPushButton::clicked, this, &MainWindow::importFace2);
	connect(ui.faceDetectionButton, &QPushButton::clicked, this, &MainWindow::startSeetaFace);
	connect(ui.faceAlignmentButton, &QPushButton::clicked, this, &MainWindow::startFaceAlignment);
	face1 = nullptr;
	face2 = nullptr;
	faceFlag = true;

	//��ʼ������ʶ��ģ��
	FDB = new seeta::FaceDatabase(seeta::ModelSetting("./model/fr_2_10.dat", seeta::ModelSetting::CPU, 0));
}

void MainWindow::importFace(int flag)
{
	//����֧��PNGͼƬ
	QString filename = QFileDialog::getOpenFileName(this, "ѡ��ͼƬ�ļ�", ".", "*.jpg *.bmp *.gif *.png");
	if (filename.isEmpty()) return;
	if (flag == 1)
	{
		this->face1 = new QImage(filename);
		ui.face1Label->setAlignment(Qt::AlignCenter);
		ui.face1Label->setPixmap(QPixmap::fromImage(*this->face1));//��ʾͼƬ
	}
	else if (flag == 2)
	{
		this->face2 = new QImage(filename);
		ui.face2Label->setAlignment(Qt::AlignCenter);
		ui.face2Label->setPixmap(QPixmap::fromImage(*this->face2));//��ʾͼƬ
	}
}

void MainWindow::importFace1()
{
	importFace(1);
}

void MainWindow::importFace2()
{
	importFace(2);
}

void MainWindow::startSeetaFace()
{
	if (this->face1 == nullptr || this->face2 == nullptr)
	{
		QMessageBox::information(this, "��ʾ��", "���ȵ�������ͼ��");
		return;
	}

	if (this->faceFlag == false)
	{
		//������ƶ�
		ui.similarityLabel->setText(QString("���ƶȣ�") + QString::number(0));
	}
	else
	{
		this->face1Mat = QImage2cvMat(*this->face1);
		this->face2Mat = QImage2cvMat(*this->face2);
		//�����ȶ�
		this->result = this->getSimilar(this->face1Mat, this->face1Points, this->face2Mat, this->face2Points);
		//������ƶ�
		ui.similarityLabel->setText(QString("���ƶȣ�") + QString::number(this->result));
	}
}

void MainWindow::startFaceAlignment()
{
	//��ʼ��
	ui.similarityLabel->setText(QString("���ƶȣ�"));

	this->faceFlag = true;

	if (this->face1 == nullptr || this->face2 == nullptr)
	{
		QMessageBox::information(this, "��ʾ��", "���ȵ�������ͼ��");
		return;
	}

	cv::Mat tempFace1 = QImage2cvMat(*this->face1);
	cv::Mat tempFace2 = QImage2cvMat(*this->face2);
	this->face2Points.clear();
	this->face1Points.clear();
	std::thread t1(&MainWindow::faceDetectionFunc, this, std::ref(tempFace1), std::ref(this->face1Points), "face_detection_face1");
	std::thread t2(&MainWindow::faceDetectionFunc, this, std::ref(tempFace2), std::ref(this->face2Points), "face_detection_face2");
	t1.join();
	t2.join();

	cv::destroyAllWindows();
	cv::imshow("face_detection_face1", tempFace1);
	cv::imshow("face_detection_face2", tempFace2);
}

//������⡢81��������
void MainWindow::faceDetectionFunc(cv::Mat& mat, std::vector<SeetaPointF>& facePoints, std::string windName)
{
	//ʹ�ö��̴߳���ͼƬ���ӿ��ٶ�
	std::cout << std::this_thread::get_id() << endl;
	seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
	//��ʼ��ģ�ͣ���Ҫע����ǣ�ģ���ļ�ͬһʱ��ֻ�ܴ�һ�Σ����Ҽ���ģ��ʱ��Ҫһ����ʱ�䣬������Ҫ���м�������
	this->myMutex.lock();
	seeta::ModelSetting FD_model("./model/fd_2_00.dat", device, 0);
	//����81�㶨λ֮�󣬵õ����������ƶȽ���������ԭ����룺
	//ͼƬ�ĳߴ�̫С������81�㶼������һ��Ȼ����������ȶ�ʱ�����
	//seeta::ModelSetting FL_model("./model/pd_2_00_pts81.dat", device, 0);
	seeta::ModelSetting FL_model("./model/pd_2_00_pts5.dat", device, 0);
	this->myMutex.unlock();
	seeta::FaceDetector FD(FD_model);
	seeta::FaceLandmarker FL(FL_model);
	//�������������С�ߴ磬����С��20
	FD.set(seeta::v2::FaceDetector::PROPERTY_MIN_FACE_SIZE, 20);
	//���ش�����ͼƬ
	seeta::cv::ImageData image = mat.clone();
	//����������⣬��ȡ���е�������Ϣ
	SeetaFaceInfoArray facesArr = FD.detect(image);
	for (int i = 0; i < facesArr.size; ++i)
	{
		//��ǰѡ�����
		auto& currentFace = facesArr.data[i];
		//�Ը�������5���81�㶨λ
		auto currentFace81Points = FL.mark(image, currentFace.pos);
		//����Ǽ�⵽�ĵ�һ������ʹ����ɫ����������ǳ���,����ʹ�ú�ɫ�������б��
		if (i == 0)
		{
			facePoints = currentFace81Points;
			cv::rectangle(mat,
				cv::Rect(currentFace.pos.x, currentFace.pos.y, currentFace.pos.width, currentFace.pos.height),
				CV_RGB(0, 0, 255));
		}
		else
		{
			cv::rectangle(mat,
				cv::Rect(currentFace.pos.x, currentFace.pos.y, currentFace.pos.width, currentFace.pos.height),
				CV_RGB(255, 0, 0));
		}
		//��81��������ʹ����ɫ���
		for (auto& point : currentFace81Points)
		{
			cv::circle(mat, cv::Point(point.x, point.y), 2, CV_RGB(0, 255, 0));
		}
	}
	cv::imwrite(windName + ".jpg", mat);

	//δ��⵽��������faceFlag==false
	if (facesArr.size == 0)
	{
		this->faceFlag = false;
	}
}

float MainWindow::getSimilar(cv::Mat& face1, std::vector<SeetaPointF>& face1Points, cv::Mat& face2, std::vector<SeetaPointF>& face2Points)
{
	seeta::cv::ImageData face1Data = face1;
	seeta::cv::ImageData face2Data = face2;
	//���֮ǰ���������
	this->FDB->Clear();
	return this->FDB->Compare(face1Data, face1Points.data(), face2Data, face2Points.data());
}

// QImageתCV::Mat
cv::Mat MainWindow::QImage2cvMat(const QImage& inImage, bool inCloneImageData)
{
	switch (inImage.format())
	{
		// 8-bit, 4 channel
	case QImage::Format_ARGB32:
	case QImage::Format_ARGB32_Premultiplied:
	{
		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC4,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		return (inCloneImageData ? mat.clone() : mat);
	}
	// 8-bit, 3 channel
	case QImage::Format_RGB32:
	{
		if (!inCloneImageData)
		{
			qWarning() << "QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
		}

		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC4,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		cv::Mat  matNoAlpha;

		cv::cvtColor(mat, matNoAlpha, cv::COLOR_BGRA2BGR);   // drop the all-white alpha channel

		return matNoAlpha;
	}
	// 8-bit, 3 channel
	case QImage::Format_RGB888:
	{
		if (!inCloneImageData)
		{
			qWarning() << "QImageToCvMat() - Conversion requires cloning so we don't modify the original QImage data";
		}

		QImage   swapped = inImage.rgbSwapped();

		return cv::Mat(swapped.height(), swapped.width(),
			CV_8UC3,
			const_cast<uchar*>(swapped.bits()),
			static_cast<size_t>(swapped.bytesPerLine())
		).clone();
	}

	// 8-bit, 1 channel
	case QImage::Format_Indexed8:
	{
		cv::Mat  mat(inImage.height(), inImage.width(),
			CV_8UC1,
			const_cast<uchar*>(inImage.bits()),
			static_cast<size_t>(inImage.bytesPerLine())
		);

		return (inCloneImageData ? mat.clone() : mat);
	}
	default:
		qWarning() << "QImage2CvMat() - QImage format not handled in switch:" << inImage.format();
		break;
	}
	return cv::Mat();
}

//CV::MatתQImage
QImage MainWindow::cvMat2QImage(const cv::Mat& inMat)
{
	switch (inMat.type())
	{
		// 8-bit, 4 channel
	case CV_8UC4:
	{
		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_ARGB32);
		return image;
	}
	// 8-bit, 3 channel
	case CV_8UC3:
	{
		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	// 8-bit, 1 channel
	case CV_8UC1:
	{
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Grayscale8);
#else
		static QVector<QRgb>  sColorTable;

		// only create our color table the first time
		if (sColorTable.isEmpty())
		{
			sColorTable.resize(256);

			for (int i = 0; i < 256; ++i)
			{
				sColorTable[i] = qRgb(i, i, i);
			}
		}

		QImage image(inMat.data,
			inMat.cols, inMat.rows,
			static_cast<int>(inMat.step),
			QImage::Format_Indexed8);

		image.setColorTable(sColorTable);
#endif
		return image;
	}
	default:
		qWarning() << "cvMat2QImage() - cv::Mat image type not handled in switch:" << inMat.type();
		break;
	}
	return QImage();
}

MainWindow::~MainWindow()
{
	if (this->face1 != nullptr) delete this->face1;
	if (this->face2 != nullptr) delete this->face2;
	if (this->FDB != nullptr)delete this->FDB;
	this->face1 = nullptr;
	this->face2 = nullptr;
	this->FDB = nullptr;
}
