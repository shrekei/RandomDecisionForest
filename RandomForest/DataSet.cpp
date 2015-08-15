#include "DataSet.h"
#include "MathUtilities.h"
using namespace std;

DataSet::DataSet()
{
	Samples = NULL;
}

DataSet::~DataSet()
{
	if (Samples != NULL)
		delete[] Samples;
}

// find the range of the feature values for each dimension
void DataSet::findFeatRange(void)
{
	FeatureRanges.resize(FeatureDim);	
	for (int i = 0; i < FeatureDim; i++) 
	{
		double fMin = _DOUBLE_MAX;
		double fMax = _DOUBLE_MIN;
		for (int j = 0; j < SampleNum; j++)
		{
			if (Samples[j].Feature[i] < fMin)
				fMin = Samples[j].Feature[i];
			if (Samples[j].Feature[i] > fMax)
				fMax = Samples[j].Feature[i];
		}
		FeatureRanges.at(i) = pair<double, double>(fMin, fMax);
	}
}

void DataSet::loadDataset(string strFileName) 
{
	FILE *pDataFile = fopen(strFileName.c_str(), "rb");
	if (!pDataFile)
	{
		cout << "Could not open input file " << strFileName << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Loading data file: " << strFileName << " ... " << endl;

	// reading the header
	fread(&SampleNum, sizeof(int), 1, pDataFile);
	fread(&FeatureDim, sizeof(int), 1, pDataFile);
	fread(&ClassNum, sizeof(int), 1, pDataFile);
	if (Samples != NULL)
		delete[] Samples;
	Samples = new Sample[SampleNum];

	// reading the training samples
	int nSampleCount = 0;
	vector<int> vecLabelCounters(ClassNum, 0);
	do
	{
		Samples[nSampleCount].Feature.resize(FeatureDim);

		// read the feature values
		for (int j = 0; j < FeatureDim; j++)
		{
			float fValue;
			if (fread(&fValue, sizeof(float), 1, pDataFile) != 1)
				break;
			Samples[nSampleCount].Feature[j] = fValue;
		}

		// read the class no. of the current sample point
		int label;
		if (fread(&label, sizeof(int), 1, pDataFile) != 1)
			break;
		Samples[nSampleCount].Label = label;
		vecLabelCounters[label]++;
		Samples[nSampleCount].Weight = 1.0;
		nSampleCount++;
		
		// check whether the maximum number of samples are read in
		if (nSampleCount >= SampleNum)
			break;
	}while (!feof(pDataFile));
	fclose(pDataFile);

	// output the number of negative and positive classes
	cout << "Label count: ";
	for (int i = 0; i < ClassNum; i++)
		cout << vecLabelCounters[i] << "	";
	cout << endl;

	// check whether the input is valid
	if (SampleNum != nSampleCount)
	{
		cout << "Could not load " << SampleNum << " samples from " << strFileName;
		cout << ". There were only " << nSampleCount << " samples!" << endl;
		SampleNum = nSampleCount;
	}

	// find the data range
	findFeatRange();

	cout << "Loaded " << nSampleCount << " samples with " << FeatureDim;
	cout << " features and " << ClassNum << " classes." << endl;
}

void DataSet::loadMultipleDatasets(vector<string> vecFileNames)
{
	// estimate the size of the combined dataset	
	SampleNum = 0;
	for (std::vector<string>::iterator itfn = vecFileNames.begin();
		itfn != vecFileNames.end(); itfn++)
	{
		FILE *pDataFile = fopen((*itfn).c_str(), "rb");
		if (!pDataFile)
		{
			std::cout << "Could not open input file " << *itfn << endl;
			exit(EXIT_FAILURE);
		}

		// read the sample number
		int nCurSampleNum;
		fread(&nCurSampleNum, sizeof(int), 1, pDataFile);
		fclose(pDataFile);
		SampleNum += nCurSampleNum;
	}
	
	// initialize the dataset storage
	if (Samples != NULL)
		delete[] Samples;
	Samples = new Sample[SampleNum];
	
	// load the multiple datasets
	int nSampleCount = 0;
	for (std::vector<string>::iterator itfn = vecFileNames.begin();
		itfn != vecFileNames.end(); itfn++)
	{
		FILE *pDataFile = fopen((*itfn).c_str(), "rb");
		if (!pDataFile)
		{
			std::cout << "Could not open input file " << *itfn << endl;
			exit(EXIT_FAILURE);
		}
		std::cout << "Loading data file: " << *itfn << " ... " << endl;

		// reading the header
		int nTemp, nCurSampleNum;
		fread(&nCurSampleNum, sizeof(int), 1, pDataFile);
		fread(&nTemp, sizeof(int), 1, pDataFile);
		if (itfn == vecFileNames.begin())
			FeatureDim = nTemp;
		else
			assert(FeatureDim == nTemp);
		fread(&nTemp, sizeof(int), 1, pDataFile);
		if (itfn == vecFileNames.begin())
			ClassNum = nTemp;
		else
			assert(ClassNum == nTemp);
		
		// reading the training samples
		int nCurSampleCount = 0;
		vector<int> vecLabelCounters(ClassNum, 0);
		do
		{
			Samples[nSampleCount].Feature.resize(FeatureDim);

			// read the feature values
			for (int j = 0; j < FeatureDim; j++)
			{
				float fValue;
				if (fread(&fValue, sizeof(float), 1, pDataFile) != 1)
					break;
				Samples[nSampleCount].Feature[j] = fValue;
			}

			// read the class no. of the current sample point
			int label;
			if (fread(&label, sizeof(int), 1, pDataFile) != 1)
				break;
			Samples[nSampleCount].Label = label;
			vecLabelCounters[label]++;
			Samples[nSampleCount].Weight = 1.0;
			nSampleCount++;
			nCurSampleCount++;
			
			// check whether the maximum number of samples are read in
			if (nCurSampleCount >= nCurSampleNum)
				break;
		}while (!feof(pDataFile));
		fclose(pDataFile);

		// output the number of each class
		cout << "Label count: ";
		for (int i = 0; i < ClassNum; i++)
			cout << vecLabelCounters[i] << "	";
		cout << endl;

		// check whether the input is valid
		if (nCurSampleNum != nCurSampleCount)
		{
			cout << "Could not load " << nCurSampleNum << " samples from " << *itfn;
			cout << ". There were only " << nCurSampleCount << " samples!" << endl;
		}
		cout << "Loaded " << nCurSampleCount << " samples with " << FeatureDim;
		cout << " features and " << ClassNum << " classes." << endl;
	}
	
	// Find the data range
	SampleNum = nSampleCount;
	findFeatRange();
}

void DataSet::generateSubsets(DataSet* pSubsets, int size, int N) const
{
	pSubsets = new DataSet[N];
	for (int i = 0; i < N; i++)
		generateSubset(pSubsets[i], size);
}

void DataSet::generateSubset(DataSet &dsSub, int size) const
{
	// init the dataset parameters
	dsSub.ClassNum = ClassNum;
	dsSub.FeatureDim = FeatureDim;
	dsSub.SampleNum = size;
	dsSub.Samples = new Sample[size];

	// perform resampling on the original dataset
	vector<int> vecRandIndex;
	randPerm(SampleNum, size, vecRandIndex);
	int nSampleCount = 0;
	for (vector<int>::iterator itri = vecRandIndex.begin(); 
		itri != vecRandIndex.end(); itri++)
	{
		dsSub.Samples[nSampleCount] = Samples[*itri];
		nSampleCount++;
	}

	// update the feature ranges
	dsSub.findFeatRange();
}