

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> labels;


	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    labels = vector<unordered_map<int, int>>(ldim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_features(const PointXYZ p, vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		return;
	}
	void update_classes(const PointXYZ p, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_points(const PointXYZ p)
	{
		count += 1;
		point += p;
		return;
	}
};

void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(vector<PointXYZ>& original_points,
                            vector<PointXYZ>& subsampled_points,
                            vector<float>& original_features,
                            vector<float>& subsampled_features,
                            vector<int>& original_classes,
                            vector<int>& subsampled_classes,
                            vector<int>& original_batches,
                            vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);

