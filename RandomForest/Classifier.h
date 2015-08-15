#pragma once
#include <vector>
#include <cxcore.h>
#include <cv.h>
#include "DataSet.h"

class Classifier
{
public:
	virtual void train(const DataSet &dataset) = 0;
	virtual Result eval(const Sample &sample) = 0;
	virtual Result eval(cv::Point p) = 0;					// evaluate a pixel using the depth context	
};
