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

#include "FeatureFPFH.h"

#include <fstream>
#include <iostream>

#include "Sample/SampleUtil.h"

#include "Data/StyleSimilarityConfig.h"

#include "Feature/FeatureUtil.h"

using namespace StyleSimilarity;

const int NUM_QUANTIZATIONS = 11;
const int NUM_BINS = NUM_QUANTIZATIONS * 3;

//#define DEBUG_OUTPUT

bool FeatureFPFH::calculate(
	TSampleSet &samples,
	TSampleSet &points,
	vector<vector<double>> &features,
	double radius)
{
#ifdef DEBUG_OUTPUT
	cout << "Building graph..." << endl;
#endif

	vector<vector<pair<int, float>>> sampleGraph; // (sample ID, distance) : # of neighbors : # of samples
	vector<vector<pair<int, float>>> pointGraph; // (point ID, distance) : # of neighbors : # of samples
	sampleGraph.resize(samples.amount);
	pointGraph.resize(samples.amount);

	SKDTree sampleTree;
	SKDTreeData sampleTreeData;
	if (!SampleUtil::buildKdTree(samples.positions, sampleTree, sampleTreeData)) return false;

	SKDTree pointTree;
	SKDTreeData pointTreeData;
	if (!SampleUtil::buildKdTree(points.positions, pointTree, pointTreeData)) return false;

#pragma omp parallel for
	for (int sampleID = 0; sampleID < samples.amount; sampleID++) {
		vec3 p = samples.positions[sampleID];
		SKDT::NamedPoint queryPoint(p[0], p[1], p[2]);

		Thea::BoundedSortedArray<SKDTree::Neighbor> sampleQueryResult(100);
		sampleTree.kClosestElements<Thea::MetricL2>(queryPoint, sampleQueryResult, radius);
		sampleGraph[sampleID].clear();
		for (int queryID = 0; queryID < sampleQueryResult.size(); queryID++) {
			int neighborID = (int)sampleTree.getElements()[sampleQueryResult[queryID].getIndex()].id;
			if (sampleID != neighborID) {
				float dist = (samples.positions[neighborID] - p).length();
				sampleGraph[sampleID].push_back(make_pair(neighborID, dist));
			}
		}

		Thea::BoundedSortedArray<SKDTree::Neighbor> pointQueryResult(100);
		pointTree.kClosestElements<Thea::MetricL2>(queryPoint, pointQueryResult, radius);
		pointGraph[sampleID].clear();
		for (int queryID = 0; queryID < pointQueryResult.size(); queryID++) {
			int neighborID = (int)pointTree.getElements()[pointQueryResult[queryID].getIndex()].id;
			float dist = (points.positions[neighborID] - p).length();
			if(dist != 0) { // should not compare sample ID and neighbor ID
				pointGraph[sampleID].push_back(make_pair(neighborID, dist));
			}
		}
	}

	// compute SPFH

#ifdef DEBUG_OUTPUT
	cout << "Computing SPFH..." << endl;
#endif

	vector<vector<double>> histSPFH(samples.amount); // SPFH : # samples
#pragma omp parallel for
	for (int sampleID = 0; sampleID < samples.amount; sampleID++) {

		vec3 sampleP = samples.positions[sampleID];
		vec3 sampleN = samples.normals[sampleID];

		vector<int> binCounts(NUM_BINS, 0);
		int totalCounts = 0;

		for (auto &neighbor : pointGraph[sampleID]) {
			int neighborID = neighbor.first;
			vec3 neighborP = points.positions[neighborID];
			vec3 neighborN = points.normals[neighborID];

			vec3 frameU = cml::normalize(sampleN);
			vec3 frameV = cml::cross(neighborP - sampleP, frameU);
			if (frameV.length_squared()) frameV.normalize();
			else continue; // cannot establish Darboux frame
			vec3 frameW = cml::normalize(cml::cross(frameU, frameV));

			float alpha = cml::dot(frameV, neighborN); // [-1, 1]
			float phi = cml::dot(frameU, cml::normalize(neighborP - sampleP)); // [-1, 1]
			float theta = atan2(cml::dot(frameW, neighborN), cml::dot(frameU, neighborN)); // [-pi, pi]

			int binPosAlpha = cml::clamp((int)(fabs(alpha)*NUM_QUANTIZATIONS), 0, NUM_QUANTIZATIONS-1);
			int binPosPhi = cml::clamp((int)(fabs(phi)*NUM_QUANTIZATIONS), 0, NUM_QUANTIZATIONS-1);
			int binPosTheta = cml::clamp((int)(fabs(theta*cml::constantsf::inv_pi())*NUM_QUANTIZATIONS), 0, NUM_QUANTIZATIONS-1);
			binCounts[binPosAlpha]++;
			binCounts[binPosPhi + NUM_QUANTIZATIONS]++;
			binCounts[binPosTheta + NUM_QUANTIZATIONS*2]++;
			totalCounts++; // not += 3
		}

		auto &hist = histSPFH[sampleID];
		hist.resize(NUM_BINS);
		for (int binID = 0; binID < NUM_BINS; binID++) {
			hist[binID] = totalCounts ? binCounts[binID] / (double)totalCounts : 0;
		}
	}

	// compute FPFH

#ifdef DEBUG_OUTPUT
	cout << "Computing FPFH..." << endl;
#endif

	features.resize(samples.amount);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < samples.amount; sampleID++) {

		auto &feature = features[sampleID];
		feature.resize(NUM_BINS, 0);
		float totalWeight = 0;
		for (auto &neighbor : sampleGraph[sampleID]) {
			int neighborID = neighbor.first;
			float weight = 1 / neighbor.second;
			for (int binID = 0; binID < NUM_BINS; binID++) {
				feature[binID] += histSPFH[neighborID][binID] * weight;
				totalWeight += weight;
			}
		}

		for (int binID = 0; binID < NUM_BINS; binID++) {
			if (totalWeight) feature[binID] /= totalWeight;
			feature[binID] += histSPFH[sampleID][binID];
		}
	}

	return true;
}