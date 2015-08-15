#pragma once
#include <cv.h>
#include <cxcore.h>
#include "DataSet.h"
#include "MathUtilities.h"

// the hypothesis is a hyperplane parallel to one dimension of the feature space
class NaiveRandomTest
{
public:
	NaiveRandomTest(void) {}
	NaiveRandomTest(int nClassNum);
	NaiveRandomTest(int nClassNum, double fFeatMin, double fFeatMax);

public:
	virtual void reset(void);
	virtual bool eval(const Sample &sample) = 0;
	virtual void update(const Sample &sample) = 0;
	std::pair<std::vector<double> , std::vector<double> > getStats(void);
	double score(void);

protected:
	int		m_nClassNum;
	double	m_fThreshold;						// threshold for feature value comparison
	double	m_fFeatMin, m_fFeatMax;			// feature value range of the feature dimension for this random test
	double	m_fTrueCount;						// number of samples going into the left child branch
	double	m_fFalseCount;					// number of samples going into the right child branch
	std::vector<double>	m_vecTrueStats;	// label distribution of samples going into the left child branch
	std::vector<double>	m_vecFalseStats;	// label distribution of samples going into the right child branch
};

// in this class, the hypothesis is an arbitrary hyperplane
class HyperPlaneTest: public NaiveRandomTest 
{
public:
	HyperPlaneTest(void) {}
	HyperPlaneTest(int nClassNum, int nFeatureDim, int nProjFeatureNum, 
		const std::vector<std::pair<double, double> > &vecFeatureRanges);

public:
	void	update(const Sample &sample);
	bool	eval(const Sample &sample);
	bool	eval(cv::Point p);	
	void	getInfo(double &fThreshold, int &nProjFeatureNum, std::vector<int> &vecProjFeatures, std::vector<double> &vecProjWeights);
	void	setInfo(double fThreshold, int nProjFeatureNum, std::vector<int> vecProjFeatures, std::vector<double> vecProjWeights);

private:
	int m_nProjFeatureNum;
	std::vector<int>			m_vecProjFeatures;
	std::vector<double>	m_vecProjWeights;
};
