#include <iostream>
#include "RandomDecisionTree.h"
#include "DepthUtilities.h"
using namespace std;

RandomDecisionTree::RandomDecisionTree(int nTreeLabel, const HyperParams &hp)
{
	m_nTreeLabel = nTreeLabel;
	m_gHP = hp;
	m_pRootNode = NULL;
	m_nNodeNum = 0;
}

RandomDecisionTree::~RandomDecisionTree() 
{
	if (m_pRootNode != NULL)
		delete m_pRootNode;
	m_vecNodes.clear();
	m_nNodeNum = 0;
}

void RandomDecisionTree::train(const DataSet &dataset) 
{
	m_vecNodes.clear();
	m_nNodeNum = 0;
	m_nClassNum = dataset.ClassNum;
	m_nFeatureDim = dataset.FeatureDim;

	// generate a set of random tests
	vector<HyperPlaneTest> vecRandomTests;
	vector<bool> vecRTActive;
	for (int i = 0; i < m_gHP.RandomTestNum; i++) 
	{
		HyperPlaneTest test(dataset.ClassNum, dataset.FeatureDim, m_gHP.ProjFeatureNum, dataset.FeatureRanges);
		vecRandomTests.push_back(test);
		vecRTActive.push_back(true);
	}

	// train the RDT classifier
	vector<int> vecSampleIndices;
	m_pRootNode = new RDTNode(0, m_gHP.MaxDepth, dataset.ClassNum, dataset.FeatureDim, m_gHP.MinCounter);
	m_vecNodes.push_back(m_pRootNode);
	for (int i = 0; i < dataset.SampleNum; i++) 
		vecSampleIndices.push_back(i);
	m_pRootNode->train(dataset, vecSampleIndices, vecRandomTests, vecRTActive, m_vecNodes);
}

Result RandomDecisionTree::eval(const Sample &sample) 
{
	return m_pRootNode->eval(sample);
}

Result RandomDecisionTree::eval(cv::Point p) 
{
	Result result = m_pRootNode->eval(p);
	int confcounter = 0;
	for (vector<double>::iterator itc = result.Confidence.begin();
		itc != result.Confidence.end(); itc++)
	{
		confcounter++;
	}
	return result;
}

/* parameter description:
mtxDepthPoints: input depth image matrix, type CV_32FC3, each cv::Vec3f element is a 3D point (x,y,z), which can be 
obtained by back projecting the original 2D depth image to the 3D space
mtxLabels: output hand part labels, including 12 hand parts and background
METHOD: per-pixel classificaton with the classification tree
*/
void RandomDecisionTree::test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels)
{
	g_mtxDepthPoints = mtxDepthPoints;

	vector<Result> results;
	for (int i = 0; i < mtxLabels.rows; i++)
	{
		for (int j = 0; j < mtxLabels.cols; j++)
		{
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == g_fPlaneDepth)
				mtxLabels.at<unsigned char>(i, j) = 255;
			else
			{
				Result result = eval(cvPoint(j, i));
				unsigned char label = result.Prediction;
				mtxLabels.at<unsigned char>(i, j) = label;
			}
		}
	}
}

// save a tree to file
void RandomDecisionTree::saveTree(FILE *pFile)
{
	m_pRootNode->saveNode(pFile);
	fprintf(pFile, "\n");
}

// load a tree from file
void RandomDecisionTree::readTree(FILE *pFile, int nClassNum, int nFeatureDim)
{
	m_nClassNum = nClassNum;
	m_nFeatureDim = nFeatureDim;
	m_pRootNode = new RDTNode(0, m_gHP.MaxDepth, m_nClassNum, m_nFeatureDim, m_gHP.MinCounter);
	m_vecNodes.push_back(m_pRootNode);
	m_pRootNode->readNode(pFile, m_vecNodes);
}