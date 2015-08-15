#pragma once
#include <stdio.h>
#include <vector>
#include "DataSet.h"
#include "HyperParams.h"
#include "RandomTest.h"

class RDTNode 
{
public:
	RDTNode(void);
	RDTNode(int nDepth, int nMaxDepth, int nClassNum, int nFeatureDim, double fMinCounter);
	RDTNode(int nDepth, int nMaxDepth, int nClassNum, int nFeatureDim, double fMinCounter,
		const std::vector<double> &vecLabelStats);
	~RDTNode();

public:
	void train(const DataSet &dataset, const vector<int> &vecSampleIndices,
		std::vector<HyperPlaneTest> &vecRandomTests, std::vector<bool> vecRTActive, 
		std::vector<RDTNode*> &vecNodes);
	Result eval(const Sample &sample);
	Result eval(cv::Point p);			

public:
	void setLabel(int nNodeLabel) { m_nNodeLabel = nNodeLabel; }
	bool isLeaf(void) { return m_bLeaf; }
	bool isRoot(void) {return m_bRoot; }
	void saveNode(FILE *pFile);
	void readNode(FILE *pFile, std::vector<RDTNode*> &vecNodes);

private:		
	bool checkSplit(const DataSet &dataset, const std::vector<int> &vecSampleIndices, const std::vector<bool> &vecRTActive);

private:
	bool			m_bLeaf;							// whether the node is a leaf
	bool			m_bRoot;							// whether the node is a root
	int				m_nDepth;							// tree level of the current node
	int				m_nNodeLabel;					// unique label to identify a tree node
	int				m_nLabel;							// predicted label at the leaf nodes
	double			m_fCounter;						// counter of the samples reaching the current node
	std::vector<double>	m_vecLabelStats;			// predicted label distribution at the current node
	HyperPlaneTest		m_tOpt;					// optimal splitting plane selected during training

private:	// these parameters are used across the whole random forest
	int				m_nClassNum;							
	int				m_nFeatureDim;			
	const int		MAX_DEPTH;
	const double	MIN_COUNTER;		

private:	// the left and right children nodes
	RDTNode*	m_pLeftChildNode;
	RDTNode*	m_pRightChildNode;
};