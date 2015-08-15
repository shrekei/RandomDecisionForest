#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

class HyperParams
{
public:
	HyperParams(void) {}
	HyperParams(const std::string& confFile);

private:
	int		ReadOneInt(std::fstream &fsConfig);

public:
	int		RandomTestNum;				// number of generated hypothesis to build the random forest
	int		ProjFeatureNum;				// number of features to determine a hyperplane for random test
	int		MinCounter;					// threshold to determine whether to split the tree node
	int		MaxDepth;					// maximum depth of the random decision tree	
	int		TreeNum;					// number of trees in the random forest
	bool	UseSoftVoting;				// flag indicating to use soft voting or not
};

inline int HyperParams::ReadOneInt(std::fstream &fsConfig)
{
	std::string line;
	getline(fsConfig, line);
	int nParam;
	std::stringstream(line) >> nParam;
	return nParam;
}

inline HyperParams::HyperParams(const std::string& confFile)
{
	std::fstream fsConfig(confFile);
	
	std::cout << "MaxDepth: " << (MaxDepth = ReadOneInt(fsConfig)) << std::endl;
	std::cout << "RandomTestNum: " << (RandomTestNum = ReadOneInt(fsConfig)) << std::endl;
	std::cout << "ProjFeatureNum: " << (ProjFeatureNum = ReadOneInt(fsConfig)) << std::endl;
	std::cout << "MinCounter: " << (MinCounter = ReadOneInt(fsConfig)) << std::endl;
	std::cout << "TreeNum: " << (TreeNum = ReadOneInt(fsConfig)) << std::endl;

	int flag;
	if ((flag = ReadOneInt(fsConfig)) == 1)
		UseSoftVoting = true;
	else
		UseSoftVoting = false;
	std::cout << "UseSoftVoting: " << UseSoftVoting << std::endl;
	fsConfig.close();
}
