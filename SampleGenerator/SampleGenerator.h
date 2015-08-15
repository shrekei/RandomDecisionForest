#include <cv.h>
#include <cxcore.h>
#include <string>
#include <stdlib.h>
#include <vector>
#include <map>
#include "../RandomForest/DepthUtilities.h"
#include "../RandomForest/TemplateParser.h"

void ReadOneFrame(std::string strDir, int nIndex, cv::Mat &mtxDepthPoints,
	int** &pLabels, int &nWidth, int &nHeight, HandParams &param);
void SelectSamples(const cv::Mat &mtxDepthPoints, cv::Rect rtOBB, int** pLabels,
	int nNumPerClass, map<int, std::vector<cv::Point> > &mapSamples);
void GenerateOneSample(const std::vector<DepthFeature> &vecFeatureIndices, cv::Point p, 
	double fDepthValue, double f, double fu, double fv, int u0, int v0, 
	const cv::Mat &mtxDepthPoints, double *pFeatureVec, int nLabel, FILE *pDestFile);
void GenerateSamples(const std::vector<DepthFeature> &vecFeatureIndices,
	double f, double fu, double fv, int u0, int v0, int nFeatureDim,
	const cv::Mat &mtxDepthPoints, std::map<int, std::vector<cv::Point> > &mapSamples, FILE *pDestFile);
