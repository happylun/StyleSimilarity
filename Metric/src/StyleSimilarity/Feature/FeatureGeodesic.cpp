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

#include "FeatureGeodesic.h"

#include <fstream>
#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"

#include "Data/StyleSimilarityConfig.h"

#include "Feature/FeatureUtil.h"

using namespace StyleSimilarity;

typedef boost::property<boost::edge_weight_t, float> EdgeProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> ProximityGraph;

FeatureGeodesic::FeatureGeodesic(TSampleSet *samples, vector<double> *features) {

	mpSamples = samples;
	mpFeatures = features;
}

FeatureGeodesic::~FeatureGeodesic() {
}

bool FeatureGeodesic::calculate() {
	
	if (!buildNeighborGraph()) return false;
	if (!calculateGeodesicDistance()) return false;

	return true;
}

bool FeatureGeodesic::buildNeighborGraph() {

	cout << "Building graph..." << endl;

	mGraph.clear();
	mGraph.resize(mpSamples->amount);

	SKDTree tree;
	SKDTreeData treeData;
	if (!SampleUtil::buildKdTree(mpSamples->positions, tree, treeData)) return false;
#pragma omp parallel for
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		vec3 p = mpSamples->positions[sampleID];
		SKDT::NamedPoint queryPoint(p[0], p[1], p[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(7);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
		for (int queryID = 0; queryID < queryResult.size(); queryID++) {
			int neighborID = (int)tree.getElements()[queryResult[queryID].getIndex()].id;
			if (sampleID != neighborID) {
				float dist = (mpSamples->positions[neighborID] - p).length();
				mGraph[sampleID].push_back(make_pair(neighborID, dist));
			}
		}
	}

	return true;
}

bool FeatureGeodesic::calculateGeodesicDistance() {

	cout << "'Boost'ing graph..." << endl;

	// build graph in boost format
	ProximityGraph *geodesicGraph = new ProximityGraph(mpSamples->amount);
	float **geoDesicDistances = new float*[mpSamples->amount];
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		geoDesicDistances[sampleID] = new float[mpSamples->amount];
		for (auto &it : mGraph[sampleID]) {
			int neighborID = it.first;
			float dist = it.second;
			boost::add_edge(sampleID, neighborID, dist, *geodesicGraph);
		}
	}

	cout << "Running Johnson's algorithm..." << endl;

	// run Johnson's algorithm
	if (!boost::johnson_all_pairs_shortest_paths(*geodesicGraph, geoDesicDistances)) return false;

	cout << "Extracting features..." << endl;

	// extract feature
	mpFeatures->resize(mpSamples->amount);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		// average geodesic distance
		double totalDistance = 0;
		int totalCount = 0;
		for (int otherID = 0; otherID < mpSamples->amount; otherID++) {
			float dist = geoDesicDistances[sampleID][otherID];
			if (dist < FLT_MAX * 0.9f) {
				totalDistance += (double)dist;
				totalCount++;
			}
		}
		(*mpFeatures)[sampleID] = totalCount ? totalDistance / totalCount : 0;
	}

	// normalize feature
	double maxDist = 0;
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		double dist = (*mpFeatures)[sampleID];
		if (dist > maxDist && dist == dist) maxDist = dist;
	}
	cout << "Max distance: " << maxDist << endl;
	if (maxDist) {
		for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
			(*mpFeatures)[sampleID] /= maxDist;
		}
	}

	// clean up
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		delete[] geoDesicDistances[sampleID];
		geoDesicDistances[sampleID] = 0;
	}
	delete[] geoDesicDistances;
	delete geodesicGraph;

	return true;
}

bool FeatureGeodesic::visualize(string fileName) {

	PlyExporter pe;

	vector<vec3i> vColors;
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		double v = (*mpFeatures)[sampleID];
		vColors.push_back(FeatureUtil::colorMapping(v));
	}

	if (!pe.addPoint(&mpSamples->positions, &mpSamples->normals, &vColors)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}
