#include "SampleGenerator.h"
#include <sys/timeb.h>
using namespace std;

int main(int argc, char* argv[])
{
	// set the configuration parameters
	int nAnchor = 10;
	int nStartFileNum = 7938;
	int nEndFileNum = 8038;
	int nSamplePerClass = 30;		// the number of pixels sampled from each hand part in each frame for training
	int nClassNum = 14;			// number of classes, including a null class and background class

	int nTestFileRate = 5;			// 80% for training and 20% for testing, you can change it for different training/testing splitting
	int nTotalFileNum = nEndFileNum - nStartFileNum + 1;
	int nSampleNum;
	if (nTotalFileNum % nTestFileRate == 0)
		nSampleNum = nSamplePerClass * (nClassNum - 1) * (nTotalFileNum - (nTotalFileNum / nTestFileRate ));
	else
		nSampleNum = nSamplePerClass * (nClassNum - 1) * (nTotalFileNum - (nTotalFileNum / nTestFileRate + 1));

	int nFeatureDim = (2 * nAnchor + 1) * (2 * nAnchor + 1) - 1;
	printf("%d\n", nSampleNum);

	// write the training file header
	std::string strTempDir = "..\\..\\Dataset";
	std::string strSubDir = "DigitGestures";
	char text[255];
	sprintf(text, "%s\\%s_%d.mat", strTempDir.c_str(), strSubDir.c_str(), nAnchor);
	FILE *pDestFile = fopen(text, "wb");
	fwrite(&nSampleNum, sizeof(int), 1, pDestFile);
	fwrite(&(nFeatureDim), sizeof(int), 1, pDestFile);
	fwrite(&nClassNum, sizeof(int), 1, pDestFile);
	
	// generate the configuration info for the depth features
	// NOTE!!: must be consistent to the parameters in the program "RandomForest"
	cv::Size sTemp(320, 240);
	SetFeatureParam(60, 10, 10, sTemp.width / 2, sTemp.height / 2, 2.0, 0.3);
	
	// choose whether to use uniformly or distance-adaptively depth context sampling
	// NOTE!!: must be consistent to the choice in the program "RandomForest"
	GenerateFeatureIndices(g_vecFeatureIndices, nFeatureDim, nAnchor, 0.10);
//	GenerateFeatureIndicesApt(g_vecFeatureIndices, nFeatureDim, nAnchor, 0.10);

	// generate the training file
	cv::Mat mtxDebug = cv::Mat(sTemp, CV_8UC3);
	cv::Mat mtxDepthPoints = cv::Mat(sTemp, CV_32FC3);
	int** pLabels = NULL;
	int nWidth, nHeight;
	HandParams param;
	map<int, vector<cv::Point> > mapSamples;

	srand ( time(NULL) );
	sprintf(text, "%s\\%s", strTempDir.c_str(), strSubDir.c_str());
	for (int i = nStartFileNum; i <= nEndFileNum; i++)
	{
		if ((i - nStartFileNum) % nTestFileRate == 0)
			continue;

		ReadOneFrame(text, i, mtxDepthPoints, pLabels, nWidth, nHeight, param);
		SelectSamples(mtxDepthPoints, param.OBB, pLabels, nSamplePerClass, mapSamples);

		// show the selected sample points
		mtxDebug.setTo(cv::Scalar(0, 0, 0));
		for (map<int, vector<cv::Point> >::iterator its = mapSamples.begin(); its != mapSamples.end(); its++)
		{
			unsigned char R, G, B;
			GetDistinctColor(its->first, R, G, B);
			for (vector<cv::Point>::iterator itps = its->second.begin(); itps != its->second.end(); itps++)
			{
				mtxDebug.at<cv::Vec3b>(itps->y, itps->x) = cv::Vec3b(B, G, R);
			}
		}
		// generate the training samples
		GenerateSamples(g_vecFeatureIndices, g_f, g_fu, g_fv, g_u0, g_v0, nFeatureDim, mtxDepthPoints, mapSamples, pDestFile);
		cv::imshow("debug", mtxDebug);
		cv::waitKey(5);
	}

	fclose(pDestFile);
	return EXIT_SUCCESS;
}
