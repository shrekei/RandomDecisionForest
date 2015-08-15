#include "RandomForest.h"
#include "DepthUtilities.h"

RandomForest::RandomForest(void)
{
	m_bUseSoftVoting = false;
	m_nTreeNum = 0;
}

RandomForest::RandomForest(const HyperParams &hp)
{
	m_bUseSoftVoting = hp.UseSoftVoting;
	m_nTreeNum = hp.TreeNum;
	m_gHP = hp;
	for (int i = 0; i < m_nTreeNum; i++)
	{
		RandomDecisionTree *pTree = new RandomDecisionTree(i, hp);
		m_vecTrees.push_back(pTree);
	}
}

RandomForest::~RandomForest() 
{
	for (vector<RandomDecisionTree*>::iterator itrdt = m_vecTrees.begin();
		itrdt != m_vecTrees.end(); itrdt++)
		delete (*itrdt);
}

void RandomForest::train(const DataSet &dataset) 
{
	m_nClassNum = dataset.ClassNum;
	m_nFeatureDim = dataset.FeatureDim;

	// train the individal tree
	for (int i = 0; i < m_nTreeNum; i++)
	{
		// resample from the original dataset to train each tree
		DataSet dsSub;
		int nSubSize = min(dataset.SampleNum / m_nTreeNum, dataset.SampleNum);
		dataset.generateSubset(dsSub, nSubSize);
		m_vecTrees.at(i)->train(dsSub);
	}
}

Result RandomForest::eval(const Sample &sample)
{
	Result result, treeResult;
	for (int i = 0; i < m_nClassNum; i++) 
		result.Confidence.push_back(0.0);

	for (int i = 0; i < m_nTreeNum; i++)
	{
		treeResult = m_vecTrees[i]->eval(sample);
		if (m_bUseSoftVoting) 
			add(treeResult.Confidence, result.Confidence);
		else 
			result.Confidence[treeResult.Prediction]++;
	}

	scale(result.Confidence, 1.0 / m_nTreeNum);
	result.Prediction = argmax(result.Confidence);
	return result;
}

// the results can be either obtained by soft voting or hard voting
// soft voting: sum of the classification scores from different trees
// hard voting: counts of the hits to each class from different trees
Result RandomForest::eval(cv::Point p)
{
	Result result, treeResult;
	for (int i = 0; i < m_nClassNum; i++) 
		result.Confidence.push_back(0.0);

	for (int i = 0; i < m_nTreeNum; i++)
	{
		treeResult = m_vecTrees[i]->eval(p);
		if (m_bUseSoftVoting) 
			add(treeResult.Confidence, result.Confidence);
		else 
			result.Confidence[treeResult.Prediction]++;
	}

	scale(result.Confidence, 1.0 / m_nTreeNum);
	result.Prediction = argmax(result.Confidence);
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
METHOD: per-pixel classificaton with the classification forest
*/
void RandomForest::test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels)
{
	g_mtxDepthPoints = mtxDepthPoints;

	vector<Result> results;
#pragma omp parallel for
	for (int i = 0; i < mtxLabels.rows; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < mtxLabels.cols; j++)
		{
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == g_fPlaneDepth)
				mtxLabels.at<unsigned char>(i, j) = 13;
			else
			{
				Result result = eval(cvPoint(j, i));
				unsigned char label = result.Prediction;
				mtxLabels.at<unsigned char>(i, j) = label;
			}
		}
	}
}

void RandomForest::saveForest(std::string strFileName)
{
	FILE *pFile = fopen(strFileName.c_str(), "w");
	int nUseSoftVoting = m_bUseSoftVoting ? 0 : 1;
	fprintf(pFile, "%d	%d %d	%d	%d	%d	%d	%d\n", m_nClassNum, m_nFeatureDim, m_gHP.TreeNum, 
		nUseSoftVoting, m_gHP.MaxDepth, m_gHP.MinCounter, m_gHP.ProjFeatureNum, m_gHP.RandomTestNum);
	for (int i = 0; i < m_nTreeNum; i++)
		m_vecTrees[i]->saveTree(pFile);

	fclose(pFile);
}

void RandomForest::readForest(std::string strFileName)
{
	FILE *pFile = fopen(strFileName.c_str(), "r");
	int nUseSoftVoting;
	fscanf(pFile, "%d	%d %d	%d	%d	%d	%d	%d\n", &m_nClassNum, &m_nFeatureDim, &m_gHP.TreeNum, 
		&nUseSoftVoting, &m_gHP.MaxDepth, &m_gHP.MinCounter, &m_gHP.ProjFeatureNum, &m_gHP.RandomTestNum);
	m_bUseSoftVoting = (nUseSoftVoting == 0) ? true : false;
	m_gHP.UseSoftVoting = m_bUseSoftVoting;
	m_nTreeNum = m_gHP.TreeNum;
	for (int i = 0; i < m_nTreeNum; i++)
	{
		RandomDecisionTree *pTree = new RandomDecisionTree(i, m_gHP);
		pTree->readTree(pFile, m_nClassNum, m_nFeatureDim);
		m_vecTrees.push_back(pTree);
	}
	fclose(pFile);
}