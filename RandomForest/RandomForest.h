#pragma once
#include "Classifier.h"
#include "DataSet.h"
#include "HyperParams.h"
#include "RandomDecisionTree.h"

class RandomForest: public Classifier
{
public:
	RandomForest(void);
	RandomForest(const HyperParams &hp);
	~RandomForest();

public:
	void	train(const DataSet &dataset);
	void	test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels);				// classify the hand parts using the depth context	
	Result	eval(const Sample &sample);
	Result	eval(cv::Point p);
	void	saveForest(std::string strFileName);
	void	readForest(std::string strFileName);

protected:
	int		m_nClassNum;
	int		m_nFeatureDim;
	bool	m_bUseSoftVoting;				// whether to use soft voting to determine the label during classification
	int		m_nTreeNum;					// number of trees
	HyperParams	m_gHP;
	std::vector<RandomDecisionTree*>	m_vecTrees;	// pointers to trees
};
