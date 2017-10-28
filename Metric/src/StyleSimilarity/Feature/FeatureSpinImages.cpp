//============================================================================
//
// This file is part of the Style Similarity project.
//
// Copyright (c) 2015 - Zhaoliang Lun (author of the code) / UMass-Amherst
//
// This is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this software.  If not, see <http://www.gnu.org/licenses/>.
//
//============================================================================

#include "FeatureSpinImages.h"

#include <fstream>
#include <iostream>

#include "Sample/SampleUtil.h"

#include "Data/StyleSimilarityConfig.h"

#include "Feature/FeatureUtil.h"

using namespace StyleSimilarity;

const int NUM_QUANTIZATIONS = 8;
const int NUM_BINS = NUM_QUANTIZATIONS * NUM_QUANTIZATIONS;

bool FeatureSpinImages::calculate(
	TSampleSet &samples,
	TSampleSet &points,
	vector<vector<double>> &features,
	double radius)
{
	
	// build KD tree

	SKDTree tree;
	SKDTreeData treeData;
	if (!SampleUtil::buildKdTree(points.positions, tree, treeData)) return false;

	// create spin images

	float maxR = (float)radius;
	float maxH = maxR / 2;
	double nnRange = maxR * 1.12; // r * sqrt(5)/2

	features.resize(samples.amount);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < samples.amount; sampleID++) {
		
		vec3 sampleP = samples.positions[sampleID];
		vec3 sampleN = samples.normals[sampleID];

		vector<int> binCounts(NUM_BINS, 0);
		int totalCounts = 0;

		SKDT::NamedPoint queryPoint((float)sampleP[0], (float)sampleP[1], (float)sampleP[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1000);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult, nnRange);
		for (int qID = 0; qID < queryResult.size(); qID++) {
			int neighborID = (int)tree.getElements()[queryResult[qID].getIndex()].id;
			if (neighborID == sampleID) continue;
			vec3 neighborP = points.positions[neighborID];
			vec3 neighborN = points.normals[neighborID];
			float h = cml::dot(neighborP - sampleP, sampleN);
			float r = sqrt((neighborP - sampleP).length_squared() - h*h);
			if (r > maxR || h > maxH) continue; // outisde image
			int rBin = cml::clamp((int)(r / maxR * NUM_QUANTIZATIONS), 0, NUM_QUANTIZATIONS-1);
			int hBin = cml::clamp((int)((h+maxH) / (maxH*2) * NUM_QUANTIZATIONS), 0, NUM_QUANTIZATIONS-1);
			int binPos = hBin*NUM_QUANTIZATIONS + rBin;
			binCounts[binPos]++;
			totalCounts++;
		}

		auto &feature = features[sampleID];
		feature.resize(NUM_BINS);
		for (int binID = 0; binID < NUM_BINS; binID++) {
			feature[binID] = totalCounts ? binCounts[binID] / (double)totalCounts : 0;
		}
	}

	return true;
}
