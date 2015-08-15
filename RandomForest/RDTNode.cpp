#include <iostream>
#include <map>
#include <omp.h>
#include "RDTNode.h"
#include "DepthUtilities.h"
using namespace std;

RDTNode::RDTNode(void) : MAX_DEPTH(0), MIN_COUNTER(0.0)
{
	m_bRoot = true;
	m_pLeftChildNode = m_pRightChildNode = NULL;
	m_nNodeLabel = -1;
}

RDTNode::RDTNode(int nDepth, int nMaxDepth, int nClassNum, int nFeatureDim, double fMinCounter) : m_nDepth(nDepth),
	MAX_DEPTH(nMaxDepth), m_nClassNum(nClassNum), m_nFeatureDim(nFeatureDim), m_bLeaf(true), 
	m_fCounter(0.0), m_nLabel(-1), MIN_COUNTER(fMinCounter)
{
	if (m_nDepth == 0)
		m_bRoot = true;
	else
		m_bRoot = false;
	for (int i = 0; i < m_nClassNum; i++)
		m_vecLabelStats.push_back(0.0);
	m_pLeftChildNode = m_pRightChildNode = NULL;
}

RDTNode::RDTNode(int nDepth, int nMaxDepth, int nClassNum, int nFeatureDim, double fMinCounter,
	const std::vector<double> &vecLabelStats) :  m_nDepth(nDepth), MAX_DEPTH(nMaxDepth), 
	m_nClassNum(nClassNum), m_nFeatureDim(nFeatureDim), m_bLeaf(true), 
	m_fCounter(0.0), m_nLabel(-1), MIN_COUNTER(fMinCounter)
{
	if (m_nDepth == 0)
		m_bRoot = true;
	else
		m_bRoot = false;
	m_fCounter = sum(vecLabelStats);
	m_vecLabelStats = vecLabelStats;
	m_nLabel = argmax(m_vecLabelStats);
	m_pLeftChildNode = m_pRightChildNode = NULL;
}

RDTNode::~RDTNode()
{
	if (!m_bLeaf)
	{
		delete m_pLeftChildNode;
		delete m_pRightChildNode;
		m_pLeftChildNode = m_pRightChildNode = NULL;
	}
}

void RDTNode::train(const DataSet &dataset, const vector<int> &vecSampleIndices,
	vector<HyperPlaneTest> &vecRandomTests, vector<bool> vecRTActive, 
	std::vector<RDTNode*> &vecNodes)
{
	if (m_bRoot)
	{
		for (vector<int>::const_iterator itsi = vecSampleIndices.begin();
			itsi != vecSampleIndices.end(); itsi++)
			m_vecLabelStats[dataset.Samples[*itsi].Label] += dataset.Samples[*itsi].Weight;
		m_fCounter = sum(m_vecLabelStats);
	}

	if (checkSplit(dataset, vecSampleIndices, vecRTActive))
	{
		// find the best random test
		int nSize = vecRTActive.size();
		double fMaxScore = _DOUBLE_MIN;
		int nOptIndex;

#pragma omp parallel for
		for (int i = 0; i < nSize; i++)
		{
			if (vecRTActive.at(i) == true)
			{
				vecRandomTests.at(i).reset();
				for (vector<int>::const_iterator itsi = vecSampleIndices.begin();
					itsi != vecSampleIndices.end(); itsi++)
					vecRandomTests.at(i).update(dataset.Samples[*itsi]);

				double fScore = vecRandomTests.at(i).score();
#pragma omp critical(Update)
				{
					if (fMaxScore < fScore)
					{
						fMaxScore = fScore;
						nOptIndex = i;
					}
				}
			}
		}
		m_tOpt = vecRandomTests.at(nOptIndex);
		m_nLabel = argmax(m_vecLabelStats);
				
		// partition the current node into left and right nodes
		pair<vector<double> , vector<double> > partitionStats = m_tOpt.getStats();
		if (sum(partitionStats.first) != 0 && sum(partitionStats.second) != 0)
		{
			m_bLeaf = false;
			vecRTActive.at(nOptIndex) = false;
			m_pRightChildNode = new RDTNode(m_nDepth + 1, MAX_DEPTH, m_nClassNum, m_nFeatureDim, MIN_COUNTER, partitionStats.first);
			m_pLeftChildNode = new RDTNode(m_nDepth + 1, MAX_DEPTH, m_nClassNum, m_nFeatureDim, MIN_COUNTER, partitionStats.second);
			vecNodes.push_back(m_pRightChildNode);
			vecNodes.push_back(m_pLeftChildNode);
			vector<int> vecLeftIndices, vecRightIndices;

			for (vector<int>::const_iterator itsi = vecSampleIndices.begin(); itsi != vecSampleIndices.end(); itsi++)
			{
				bool bFlag = m_tOpt.eval(dataset.Samples[*itsi]);
				if (bFlag)
					vecRightIndices.push_back(*itsi);
				else
					vecLeftIndices.push_back(*itsi);
			}
			m_pRightChildNode->train(dataset, vecRightIndices, vecRandomTests, vecRTActive, vecNodes);
			m_pLeftChildNode->train(dataset, vecLeftIndices, vecRandomTests, vecRTActive, vecNodes);
		}
		else
		{
			m_bLeaf = true;
		}
	}
	else
	{
		m_bLeaf = true;
	}
	assert(sum(m_vecLabelStats) == m_fCounter);
}

Result RDTNode::eval(const Sample &sample) 
{
	if (m_bLeaf) 
	{
		Result result;
		if (m_fCounter != 0.0)
		{
			result.Confidence = m_vecLabelStats;
			scale(result.Confidence, 1.0 / m_fCounter);
			result.Prediction = m_nLabel;
		}
		else
		{
			for (int i = 0; i < m_nClassNum; i++)
			{
				result.Confidence.push_back(1.0 / m_nClassNum);
			}
			result.Prediction = -1;
		}
		return result;
	}
	else 
	{
		if (m_tOpt.eval(sample))
			return m_pRightChildNode->eval(sample);
		else 
			return m_pLeftChildNode->eval(sample);
	}
}

// evaluate using the depth context
Result RDTNode::eval(cv::Point p) 
{
	if (m_bLeaf) 
	{
		Result result;
		if (m_fCounter != 0.0)
		{
			result.Confidence = m_vecLabelStats;
			scale(result.Confidence, 1.0 / m_fCounter);
			result.Prediction = m_nLabel;
		}
		else
		{
			for (int i = 0; i < m_nClassNum; i++)
			{
				result.Confidence.push_back(1.0 / m_nClassNum);
			}
			result.Prediction = 0;
		}
		return result;
	}
	else 
	{
		bool bFlag = m_tOpt.eval(p);
		if (bFlag)
			return m_pRightChildNode->eval(p);
		else 
			return m_pLeftChildNode->eval(p);
	}
}

bool RDTNode::checkSplit(const DataSet &dataset, const vector<int> &vecSampleIndices, 
	const vector<bool> &vecRTActive) 
{
	// check if there is available random tests
	bool isAvailable = false;
	for (vector<bool>::const_iterator itrt = vecRTActive.begin(); 
		itrt != vecRTActive.end(); itrt++)
	{
		if (*itrt == true)
			isAvailable = true;
	}

	// check whether the current node contains only one label
	bool isPure = false;
	for (int i = 0; i < m_nClassNum; i++) 
	{
		if (m_vecLabelStats[i] == sum(m_vecLabelStats)) 
		{
			isPure = true;
			break;
		}
	}

	// the node will not be splitted if the node contains only one label or 
	// the max depth is reached or not enough samples are seen
	if (!isAvailable || isPure || m_nDepth >= MAX_DEPTH || 
		m_fCounter < MIN_COUNTER)
		return false;
	else
		return true;
}

void RDTNode::saveNode(FILE *pFile)
{
	if (m_bLeaf == false)
	{
		fprintf(pFile, "1	");
		fprintf(pFile, "%d	", m_nLabel);
		double fThreshold;
		int nProjNum;
		std::vector<int> vecProjFeatures;
		std::vector<double> vecProjWeights;
		m_tOpt.getInfo(fThreshold, nProjNum, vecProjFeatures, vecProjWeights);
		fprintf(pFile, "%lf	%d	", fThreshold, nProjNum);
		for (int i = 0; i < nProjNum; i++)
		{
			int nFeatureIndex = vecProjFeatures.at(i);
			fprintf(pFile, "%d	%lf	", nFeatureIndex, vecProjWeights.at(i));
		}
		fprintf(pFile, "\n");

		m_pLeftChildNode->saveNode(pFile);
		m_pRightChildNode->saveNode(pFile);
	}
	else
	{
		fprintf(pFile, "0	");
		fprintf(pFile, "%d	", m_nLabel);
		for (vector<double>::iterator itls = m_vecLabelStats.begin();
			itls != m_vecLabelStats.end(); itls++)
			fprintf(pFile, "%lf	", *itls);
		fprintf(pFile, "\n");
	}
}

void RDTNode::readNode(FILE *pFile, vector<RDTNode*> &vecNodes)
{
	int nLeaf;
	fscanf(pFile, "%d	", &nLeaf);
	if (nLeaf == 1)
	{	
		m_bLeaf = false;
		fscanf(pFile, "%d	", &m_nLabel);

		double fThreshold;
		int nProjNum;
		std::vector<int> vecProjFeatures;
		std::vector<double> vecProjWeights;
		fscanf(pFile, "%lf	%d	", &fThreshold, &nProjNum);
		for (int i = 0; i < nProjNum; i++)
		{
			int nProjFeature;
			double fWeight;
			fscanf(pFile, "%d	%lf	", &nProjFeature, &fWeight);
			vecProjFeatures.push_back(nProjFeature);
			vecProjWeights.push_back(fWeight);
		}
		m_tOpt.setInfo(fThreshold, nProjNum, vecProjFeatures, vecProjWeights);

		m_pRightChildNode = new RDTNode(m_nDepth + 1, MAX_DEPTH, m_nClassNum, m_nFeatureDim, MIN_COUNTER);
		m_pLeftChildNode = new RDTNode(m_nDepth + 1, MAX_DEPTH, m_nClassNum, m_nFeatureDim, MIN_COUNTER);
		vecNodes.push_back(m_pRightChildNode);
		vecNodes.push_back(m_pLeftChildNode);
		m_pLeftChildNode->readNode(pFile, vecNodes);
		m_pRightChildNode->readNode(pFile, vecNodes);
	}
	else
	{
		m_bLeaf = true;
		fscanf(pFile, "%d	", &m_nLabel);
		vector<double> vecLabelStats;
		for (int i = 0; i < m_nClassNum; i++)
		{
			double fStat;
			fscanf(pFile, "%lf	", &fStat);
			vecLabelStats.push_back(fStat);
		}
		m_vecLabelStats = vecLabelStats;
		m_fCounter = sum(m_vecLabelStats);
		m_pLeftChildNode = m_pRightChildNode = NULL;
	}
}