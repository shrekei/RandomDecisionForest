#include "TemplateParser.h"
#include <stdio.h>
#include "DepthUtilities.h"

// save the hand parameters, depth points and label images of the whole frame
void LoadImages(std::string strDir, int nIndex, cv::Mat &mtxDepthPoints,
	int** &pLabels, int &nWidth, int &nHeight, HandParams &param)
{
	char text[255];
	sprintf(text, "%s\\depth3d_%d.rhd", strDir.c_str(), nIndex);
	LoadDepthPoints(mtxDepthPoints, text);	
	sprintf(text, "%s\\label_%d.rhd", strDir.c_str(), nIndex);
	LoadLabelImage(pLabels, nWidth, nHeight, text);
	sprintf(text, "%s\\para_%d.txt", strDir.c_str(), nIndex);
	LoadHandParams(param, text);
}

void LoadDepthPoints(cv::Mat &mtxDepthPoints, char *strName)
{
	FILE *pFile = fopen(strName, "rb");
	int nWidth, nHeight;
	fread(&nWidth, sizeof(int), 1, pFile);
	fread(&nHeight, sizeof(int), 1, pFile);
	if (mtxDepthPoints.cols != nWidth || mtxDepthPoints.rows != nHeight)
	{
		mtxDepthPoints.release();
		mtxDepthPoints.create(cv::Size(nWidth, nHeight), CV_32FC3);
	}
	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
		{
			float x, y, z;
			fread(&x, sizeof(float), 1, pFile);
			fread(&y, sizeof(float), 1, pFile);
			fread(&z, sizeof(float), 1, pFile);
			mtxDepthPoints.at<cv::Vec3f>(i, j) = cv::Vec3f(x, y, z);
		}
	}
	fclose(pFile);
}

void LoadLabelImage(int** &pLabels, int &nWidth, int &nHeight, char *strName)
{
	FILE *pFile = fopen(strName, "rb");
	fread(&nWidth, sizeof(int), 1, pFile);
	fread(&nHeight, sizeof(int), 1, pFile);
	bool bInit = (pLabels == NULL);
	if (bInit)
		pLabels = new int*[nHeight];
	for (int i = 0; i < nHeight; i++)
	{
		if (bInit)
			pLabels[i] = new int[nWidth];
		for (int j = 0; j < nWidth; j++)
		{
			int nLabel;
			fread(&nLabel, sizeof(int), 1, pFile);
			pLabels[i][j] = nLabel;
		}
	}
	fclose(pFile);
}

void LoadHandParams(HandParams &param, char *strName)
{
	FILE *pFile = fopen(strName, "r");

	// write the hand region
	cv::Rect rtOBB;
	fscanf(pFile, "%d, %d, %d, %d\n", &rtOBB.x, &rtOBB.y, &rtOBB.width, &rtOBB.height);
	param.OBB = rtOBB;

	// write the pose parameters
	int nDofNum;
	fscanf(pFile, "%d\n", &nDofNum);
	std::vector<DoF> vecDofSet;
	for (int i = 0; i < nDofNum; i++)
	{
		char strDofName[255];
		int nType;
		double fValue;
		fscanf(pFile, "%s	%d	%lf\n", strDofName, &nType, &fValue);
		vecDofSet.push_back(DoF(strDofName, (DofType)nType, fValue));
	}
	param.DofSet = vecDofSet;

	// write the projected palm center and the fingertips
	int m, n;
	double x, y, z;
	fscanf(pFile, "%d, %d, %lf, %lf, %lf\n", &m, &n, &x, &y, &z);
	param.PalmCenter2D = cv::Point(m, n);
	param.PalmCenter3D = cv::Vec3f(x, y, z);
	for (int i = 0; i < 5; i++)
	{
		fscanf(pFile, "%d, %d, %lf, %lf, %lf\n", &m, &n, &x, &y, &z);
		param.Fingertips2D[i] = cv::Point(m, n);
		param.Fingertips3D[i] = cv::Vec3f(x, y, z);
	}

	fclose(pFile);
}