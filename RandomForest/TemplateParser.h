#include <cv.h>
#include <cxcore.h>
#include <string>

enum DofType {ROTATION_X, ROTATION_Y, ROTATION_Z, TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z, NOT_A_DOF};

class DoF
{
public:
	DoF() {}
	DoF(std::string name, DofType type, double value) : Name(name), Type(type), Value(value) {}

public:
	std::string	Name;
	DofType	Type;
	double		Value;
};

class HandParams
{
public:
	cv::Rect			OBB;
	std::vector<DoF>	DofSet;
	cv::Point			PalmCenter2D;
	cv::Vec3f			PalmCenter3D;
	cv::Point			Fingertips2D[5];
	cv::Vec3f			Fingertips3D[5];
};

void LoadImages(std::string strDir, int nIndex, cv::Mat &mtxDepthPoints,
	int** &pLabels, int &nWidth, int &nHeight, HandParams &param);
void LoadDepthPoints(cv::Mat &mtxDepthPoints, char *strName);
void LoadLabelImage(int** &pLabels, int &nWidth, int &nHeight, char *strName);
void LoadHandParams(HandParams &param, char *strName);

// generate a color configuration with the input index
inline void GetDistinctColor(int n, unsigned char &R, unsigned char &G, unsigned char &B)
{
	n = abs(n) % 56;
	static int nColourValues[56] = 
	{ 
		0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF, 0x800000, 
		0x008000, 0x000080, 0x808000, 0x800080, 0x008080, 0x808080, 0xC00000, 
		0x00C000, 0x0000C0, 0xC0C000, 0xC000C0, 0x00C0C0, 0xC0C0C0, 0x004040, 
		0x400000, 0x004000, 0x000040, 0x404000, 0x400040, 0x404040, 0x200000, 
		0x002000, 0x000020, 0x202000, 0x200020, 0x002020, 0x202020, 0x600000, 
		0x006000, 0x000060, 0x606000, 0x600060, 0x006060, 0x606060, 0xA00000, 
		0x00A000, 0x0000A0, 0xA0A000, 0xA000A0, 0x00A0A0, 0xA0A0A0, 0xE00000, 
		0x00E000, 0x0000E0, 0xE0E000, 0xE000E0, 0x00E0E0, 0xE0E0E0, 0x000000
	};
	R = (nColourValues[n] & 0xFF0000) >> 16;
	G = (nColourValues[n] & 0x00FF00) >> 8;
	B = nColourValues[n] & 0x0000FF;
}

// the synthesized hand consists of 30 parts, which are merged into 12 parts for classification
inline int MapToSampleLabel(int nLabel)
{
	static std::vector<std::vector<int> > vecLabels;
	if (vecLabels.size() == 0)
	{
		int nSet0[5] = {0, 18, 24, 6, 12};
		vecLabels.push_back(std::vector<int>(nSet0, nSet0 + 5));		// null hand part
		int nSet1[3] = {25, 19, 1};
		vecLabels.push_back(std::vector<int>(nSet1, nSet1 + 3));		// palm part 1
		int nSet2[2] = {7, 13};
		vecLabels.push_back(std::vector<int>(nSet2, nSet2 + 2));		// palm part 2
		int nSet3[1] = {26};
		vecLabels.push_back(std::vector<int>(nSet3, nSet3 + 1));		// thumb part 1
		int nSet4[2] = {27, 28};
		vecLabels.push_back(std::vector<int>(nSet4, nSet4 + 2));		// thumb part 2
		int nSet5[2] = {20, 21};
		vecLabels.push_back(std::vector<int>(nSet5, nSet5 + 2));		// index part 1
		int nSet6[2] = {22, 23};
		vecLabels.push_back(std::vector<int>(nSet6, nSet6 + 2));		// index part 2
		int nSet7[2] = {2, 3};
		vecLabels.push_back(std::vector<int>(nSet7, nSet7 + 2));		// middle part 1
		int nSet8[2] = {4, 5};
		vecLabels.push_back(std::vector<int>(nSet8, nSet8 + 2));		// middle part 2
		int nSet9[2] = {8, 9};
		vecLabels.push_back(std::vector<int>(nSet9, nSet9 + 2));		// ring part 1
		int nSet10[2] = {10, 11};
		vecLabels.push_back(std::vector<int>(nSet10, nSet10 + 2));	// ring part 2
		int nSet11[2] = {14, 15};
		vecLabels.push_back(std::vector<int>(nSet11, nSet11 + 2));	// pinky part 1
		int nSet12[2] = {16, 17};
		vecLabels.push_back(std::vector<int>(nSet12, nSet12 + 2));	// pinky part 2
	}

	// background label
	if (nLabel == 255)
		return vecLabels.size();

	int nCount = 0;
	for (std::vector<std::vector<int> >::iterator itl = vecLabels.begin();
		itl != vecLabels.end(); itl++)
	{
		for (std::vector<int>::iterator itsl = itl->begin(); itsl != itl->end(); itsl++)
		{
			if (*itsl == nLabel)
				return nCount;
		}
		nCount++;
	}
	return -1;
}