

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
	unordered_map<int, int> labels;


	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	}

	// Method Update
	void update_all(const PointXYZ p, std::vector<float>::iterator f_begin, const int l)
	{
		count += 1;
		point += p;
		std::transform (features.begin(), features.end(), f_begin, features.begin(), std::plus<float>());
		labels[l] += 1;
		return;
	}
	void update_features(const PointXYZ p, std::vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		std::transform (features.begin(), features.end(), f_begin, features.begin(), std::plus<float>());
		return;
	}
	void update_classes(const PointXYZ p, const int l)
	{
		count += 1;
		point += p;
		labels[l] += 1;
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
                      float sampleDl);


void batch_grid_subsampling(vector<PointXYZ>& original_points,
                              vector<PointXYZ>& subsampled_points,
                              vector<float>& original_features,
                              vector<float>& subsampled_features,
                              vector<int>& original_classes,
                              vector<int>& subsampled_classes,
                              vector<int>& original_batches,
                              vector<int>& subsampled_batches,
                              float sampleDl);

