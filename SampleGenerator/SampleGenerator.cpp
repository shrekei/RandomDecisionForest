#include <highgui.h>
#include <time.h>
#include <iostream>
#include "SampleGenerator.h"
using namespace std;

void ReadOneFrame(std::string strDir, int nIndex, cv::Mat &mtxDepthPoints,
	int** &pLabels, int &nWidth, int &nHeight, HandParams &param)
{
	LoadImages(strDir, nIndex, mtxDepthPoints, pLabels, nWidth, nHeight, param);
	static cv::Mat mtxLabels = cv::Mat(mtxDepthPoints.rows, mtxDepthPoints.cols, CV_8UC3);
	static cv::Mat mtxDepthVis = cv::Mat(mtxDepthPoints.rows, mtxDepthPoints.cols, CV_8UC3);
	const double MIN_DEPTH = 0.4;
	const double MAX_DEPTH = 1.0;
	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
		{
			pLabels[i][j] = MapToSampleLabel(pLabels[i][j]);
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == 0.0)	
			{
				mtxDepthPoints.at<cv::Vec3f>(i, j)[2] = g_fPlaneDepth;
				mtxDepthVis.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
			else
			{
				double fGray = (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255;
				mtxDepthVis.at<cv::Vec3b>(i, j) = cv::Vec3b(fGray, fGray, fGray);
			}

			unsigned char R, G, B;
			GetDistinctColor(pLabels[i][j], R, G, B);
			mtxLabels.at<cv::Vec3b>(i, j) = cv::Vec3b(R, G, B);
		}
	}
	for (int i = 0; i < 5; i++)
		cv::circle(mtxLabels, param.Fingertips2D[i], 3, cv::Scalar(255, 255, 255), 2);
	cv::circle(mtxLabels, param.PalmCenter2D, 3, cv::Scalar(255, 255, 255), 2);
	cv::rectangle(mtxLabels, cv::Point(param.OBB.x, param.OBB.y), 
		cv::Point(param.OBB.x + param.OBB.width, param.OBB.y + param.OBB.height), cv::Scalar(255, 255, 255), 1);
	cv::imshow("Label", mtxLabels);
	cv::imshow("Depth", mtxDepthVis);
}

void SelectSamples(const cv::Mat &mtxDepthPoints, cv::Rect rtOBB, int** pLabels,
	int nNumPerClass, map<int, vector<cv::Point> > &mapSamples)
{	
	// get the candidates
	vector<cv::Point> mapCandidates[14];
	for (int i = rtOBB.y; i < rtOBB.y + rtOBB.height; i++)
	{
		for (int j = rtOBB.x; j < rtOBB.x + rtOBB.width; j++)
			mapCandidates[pLabels[i][j]].push_back(cv::Point(j, i));
	}

	// generate the samples
	mapSamples.clear();
	for (int i = 0; i < 14; i++)
	{
		if (mapCandidates[i].size() < nNumPerClass || i == 0)
			continue;

		vector<cv::Point> vecCur;
		for (int j = 0; j < nNumPerClass; j++)
		{
			int nIndex = (1.0 * rand() / RAND_MAX) * (mapCandidates[i].size() - 1);
			vecCur.push_back(mapCandidates[i].at(nIndex));
		}
		mapSamples[i] = vecCur;
	}
}

void GenerateOneSample(const vector<DepthFeature> &vecFeatureIndices, cv::Point p, 
	double fDepthValue, double f, double fu, double fv, int u0, int v0, 
	const cv::Mat &mtxDepthPoints, float *pFeatureVec, int nLabel, FILE *pDestFile)
{
	int nFeatureDim = vecFeatureIndices.size();
	for (int i = 0; i < nFeatureDim; i++)
	{
		pFeatureVec[i] = CalcFeatureValue(vecFeatureIndices.at(i), p, fDepthValue, f, fu, fv, mtxDepthPoints);
	}
	fwrite(pFeatureVec, sizeof(float), nFeatureDim, pDestFile);
	fwrite(&nLabel, sizeof(int), 1, pDestFile);
}

void GenerateSamples(const vector<DepthFeature> &vecFeatureIndices,
	double f, double fu, double fv, int u0, int v0, int nFeatureDim,
	const cv::Mat &mtxDepthPoints, map<int, vector<cv::Point> > &mapSamples, FILE *pDestFile)
{
	float *pFeatureVec = new float[nFeatureDim];
	for (map<int, vector<cv::Point> >::iterator its = mapSamples.begin(); its != mapSamples.end(); its++)
	{
		for (vector<cv::Point>::iterator itp = its->second.begin(); itp != its->second.end(); itp++)
		{
			double fDepthValue = mtxDepthPoints.at<cv::Vec3f>(itp->y, itp->x)[2];
			GenerateOneSample(vecFeatureIndices,  *itp, fDepthValue, f, fu, fv, u0, v0, 
				mtxDepthPoints, pFeatureVec, its->first, pDestFile);
		}
	}
	delete[] pFeatureVec;
}
