#include "DepthUtilities.h"

double		g_f, g_fu, g_fv;
int			g_u0, g_v0;
double		g_fPlaneDepth;
double		g_fMaxDepthDiff;
cv::Mat		g_mtxDepthPoints;
std::vector<DepthFeature>	g_vecFeatureIndices;
