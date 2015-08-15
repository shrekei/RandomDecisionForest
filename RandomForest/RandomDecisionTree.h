#pragma once
#include "Classifier.h"
#include "DataSet.h"
#include "HyperParams.h"
#include "RDTNode.h"

class RandomDecisionTree: public Classifier
{
public:
	RandomDecisionTree(int nTreeLabel, const HyperParams &hp);
	~RandomDecisionTree();

public:
	void	train(const DataSet &dataset);
	void	test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels);				// classify the hand parts using the depth context	
	Result	eval(const Sample &sample);
	Result	eval(cv::Point p);
	void	saveTree(FILE *pFile);
	void	readTree(FILE *pFile, int nClassNum, int nFeatureDim);

private:
	int		m_nClassNum;
	int		m_nFeatureDim;
	int		m_nTreeLabel;							// no. of the current tree
	int		m_nNodeNum;							// node number
	HyperParams				m_gHP;			// tree parameters
	RDTNode*				m_pRootNode;	// pointer to the root node
	std::vector<RDTNode*>	m_vecNodes;		// all the tree nodes
};
