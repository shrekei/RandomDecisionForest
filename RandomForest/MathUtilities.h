#pragma once
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <assert.h>
#ifndef WIN32
#include <sys/time.h>
#endif

using namespace std;

// Random Numbers Generators
unsigned int getDevRandom();

//! Returns a random number in [0, 1]
inline double randDouble()
{
	static bool didSeeding = false;

	if (!didSeeding) 
	{
#ifdef WIN32
	//	srand (time(NULL));
		srand (0);
#else
		unsigned int seedNum;
		struct timeval TV;
		unsigned int curTime;

		gettimeofday(&TV, NULL);
		curTime = (unsigned int) TV.tv_usec;
		seedNum = (unsigned int) time(NULL) + curTime + getpid() + getDevRandom();

		srand(seedNum);
#endif
		didSeeding = true;
	}
	return rand() / (RAND_MAX + 1.0);
}

//! Returns a random number in [min, max]
inline double randomFromRange(const double &minRange, const double &maxRange)
{
	return minRange + (maxRange - minRange) * randDouble();
}

//! Random permutations
void randPerm(const int &inNum, vector<int> &outVect);
void randPerm(const int &inNum, const int inPart, vector<int> &outVect);

// fill a vector with random numbers within [-1, 1]
inline void fillWithRandomNumbers(const int &length, vector<double> &inVect) 
{
	inVect.clear();
	for (int i = 0; i < length; i++) 
	{
		inVect.push_back(2.0 * (randDouble() - 0.5));
	}
}

// return the index of the max element in a vector
inline int argmax(const vector<double> &inVect) 
{
	double maxValue = inVect[0];
	int maxIndex = 0, i = 1;
	vector<double>::const_iterator itr(inVect.begin() + 1), end(inVect.end());
	while (itr != end) 
	{
		if (*itr > maxValue) 
		{
			maxValue = *itr;
			maxIndex = i;
		}
		++i;
		++itr;
	}
	return maxIndex;
}

// sum all elements in the vector
inline double sum(const vector<double> &inVect)
{
	double val = 0.0;
	vector<double>::const_iterator itr(inVect.begin()), end(inVect.end());
	while (itr != end)
	{
		val += *itr;
		++itr;
	}
	return val;
}

//! Poisson sampling
inline int poisson(double A) 
{
	int k = 0;
	int maxK = 10;
	while (1)
	{
		double U_k = randDouble();
		A *= U_k;
		if (k > maxK || A < exp(-1.0))
		{
			break;
		}
		k++;
	}
	return k;
}

inline double log2(double p)
{
	return log10(p) / log10(2.0);
}

inline void add(std::vector<double> lv, std::vector<double> &rv)
{
	assert(lv.size() == rv.size());
	int len = lv.size();
	for (int i = 0; i < len; i++)
	{
		rv.at(i) += lv.at(i);
	}
}

inline void scale(std::vector<double> &lv, double ratio)
{
	int len = lv.size();
	for (int i = 0; i < len; i++)
	{
		lv.at(i) *= ratio;
	}
}