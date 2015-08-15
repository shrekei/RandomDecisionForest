#pragma once
#include <vector>
#include <math.h>
#include <cxcore.h>
#include <highgui.h>
#include <cv.h>

// depth feature: a pair of neighbor points
class DepthFeature
{
public:
	DepthFeature(void) {}
	DepthFeature(cv::Point2f  u0) : u(u0){}
	cv::Point2f u;
	double theta;
};

// parameters and functions used to calculate the depth features
// see function "SetFeatureParam" for the parameter definitions
extern double	g_f, g_fu, g_fv;
extern int		g_u0, g_v0;
extern double	g_fPlaneDepth;
extern double	g_fMaxDepthDiff;
extern cv::Mat	g_mtxDepthPoints;
extern std::vector<DepthFeature>	g_vecFeatureIndices;

inline void SetFeatureParam(double f0, double fu0, double fv0,
	int u00, int v00, double fPlaneDepth, double fMaxDepthDiff)
{
	g_f = f0;			// the focal length in millimeter
	g_fu = fu0;			// pixel number along x on the image plane within 1 millimeter
	g_fv = fv0;			// pixel number along y on the image plane within 1 millimeter
	g_u0 = u00;		// the center x in image coordinate
	g_v0 = v00;		// the center y in image coordinate
	g_fPlaneDepth = fPlaneDepth;			// the default depth value for background, by which we assume a constant depth value for the background if hand is segmented
	g_fMaxDepthDiff = fMaxDepthDiff;	// maximum depth difference in the depth context feature
}

// UNIFORMLY generate the indices of the neighboring context points for depth context
// please refer to Section V.A of our paper Hui Liang, Junsong Yuan, and Daniel Thalmann, 
// Parsing the Hand in Depth Images, in IEEE Trans. Multimedia, 2014, for details
//---------------------------------parameters------------------------------------------//
// vecFeatureIndices: the coordinates of the depth context points
// nAnchor: the number M to indicate number of context points in our paper
// fRadius: size of the depth context in METER
inline void GenerateFeatureIndices(std::vector<DepthFeature> &vecFeatureIndices,
	int nFeatureDim, int nAnchor, double fRadius)
{
	int nGridPointNum = (2 * nAnchor + 1) * (2 * nAnchor + 1) - 1;
	assert(nGridPointNum == nFeatureDim );
	
	// generate the grid points
	vecFeatureIndices.resize(nFeatureDim);
	int nFeatCount = 0;
	for (int i = -nAnchor; i <= nAnchor; i++)
	{
		for (int j = -nAnchor; j <= nAnchor; j++)
		{
			if (i == 0 && j == 0)
				continue;
			double ux = 1.0 * j / nAnchor * fRadius;
			double uy = 1.0 * i / nAnchor * fRadius ;
			DepthFeature dfTmp(cv::Point2f(ux, uy));
			vecFeatureIndices.at(nFeatCount++) = dfTmp;
		}
	}
}

// DISTANCE-ADAPTIVE scheme to generate the indices of the neighboring context points for depth context
// please refer to Section V.A of our paper Hui Liang, Junsong Yuan, and Daniel Thalmann, 
// Parsing the Hand in Depth Images, in IEEE Trans. Multimedia, 2014, for details
//---------------------------------parameters------------------------------------------//
// vecFeatureIndices: the coordinates of the depth context points
// nAnchor: the number M to indicate number of context points in our paper
// fRadius: size of the depth context in METER
inline void GenerateFeatureIndicesApt(std::vector<DepthFeature> &vecFeatureIndices,
	int nFeatureDim, int nAnchor, double fRadius)
{
	int nGridPointNum = (2 * nAnchor + 1) * (2 * nAnchor + 1) - 1;
	assert(nGridPointNum == nFeatureDim );
	
	// generate the grid points
	double c0, k;
	double fRatio = 0.2;
	c0 = nAnchor / (fRadius + (fRatio - 1) * fRadius / 2);
	k = (fRatio - 1) * c0 / fRadius;
	vecFeatureIndices.resize(nFeatureDim);
	int nFeatCount = 0;
	for (int i = -nAnchor; i <= nAnchor; i++)
	{
		for (int j = -nAnchor; j <= nAnchor; j++)
		{
			double si, sj, vi, vj;
			si = i > 0 ? 1 : -1;
			sj = j > 0 ? 1 : -1;
			vi = fabs(1.0 * i);
			vj = fabs(1.0 * j);
			if (i == 0 && j == 0)
				continue;
			double ux = sj * (-c0 + sqrt(c0 * c0 + 4 * vj * k / 2)) / k;
			double uy = si * (-c0 + sqrt(c0 * c0 + 4 * vi * k / 2)) / k;
			DepthFeature dfTmp(cv::Point2f(ux, uy));
			vecFeatureIndices.at(nFeatCount++) = dfTmp;
		}
	}
}

inline double CalcFeatureValue(DepthFeature ftIn, cv::Point p, double fDepthValue, 
	double f, double fu, double fv, const cv::Mat &mtxDepthPoints)
{
	if (fDepthValue == g_fPlaneDepth)
		return 0.0;

	cv::Point ud;
	ud.x = ftIn.u.x / fDepthValue * f * fu + p.x;
	ud.y = ftIn.u.y / fDepthValue * f * fv + p.y;
	
	double fDepthU;
	if (ud.x < 0 || ud.x >= mtxDepthPoints.cols || ud.y < 0 || ud.y >= mtxDepthPoints.rows)
		fDepthU = g_fPlaneDepth;						// assumed to be background
	else
		fDepthU = mtxDepthPoints.at<cv::Vec3f>(ud.y, ud.x)[2];
	
	double fFeatureValue = fDepthU - fDepthValue; 
	if (fabs(fFeatureValue) > g_fMaxDepthDiff)
	{
		double fSign = fFeatureValue >= 0 ? 1 : -1;
		fFeatureValue = fSign * g_fMaxDepthDiff;
	}
	return fFeatureValue;
}
