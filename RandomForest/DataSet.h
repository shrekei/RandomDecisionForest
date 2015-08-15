#pragma once
#include <vector>
#include <string>

// commonly used quantities
const double RADTODEG =  57.2958;	// 360 / (2 * PI)
const double PI = 3.14159265359;

// the maximum values
const double _DOUBLE_MAX = 1.7e+308;
const double _DOUBLE_MIN = -1.7e+308;
const short _SHORT_MAX = 32767;
const short _SHORT_MIN = -32768;
const int _INT_MIN = -2147483648;
const int _INT_MAX = 2147483647;

class Sample
{
public:	
	std::vector<float>	Feature;		// feature vector
	int					Label;			// annotated label for classification
	double				Weight;			// prior for different classes, i.e. the samples of different classes can be unbalanced. Use this term to balance them
};

class Result
{
public:
	std::vector<double>	Confidence;	// predicted label distribution
	int						Prediction;		// prediction = argmax Confidence
};

class DataSet 
{
public:
	DataSet();
	virtual ~DataSet();

public:
	void		findFeatRange(void);
	void		loadDataset(std::string strFileName);
	void		loadMultipleDatasets(std::vector<std::string> vecFileNames);
	void		generateSubsets(DataSet* pSubsets, int size, int N) const;
	void		generateSubset(DataSet &dsSub, int size) const;

private:
	DataSet(const DataSet&);
	DataSet& operator=(const DataSet&);

public:
	Sample*	Samples;					// all the samples for training
	int			SampleNum;				// number of samples
	int			FeatureDim;				// feature dimension
	int			ClassNum;					// number of classes for classification
	std::vector<std::pair<double, double> >	FeatureRanges;		// min and max values for each dimension of the feature vector for all the samples
};

