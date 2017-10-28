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

#include "SegmentSampleSpectral.h"

#include <fstream>
#include <set>

#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"
#include "Segment/SegmentUtil.h"

#include "Cluster/ClusterSparseSpectral.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

#define OUTPUT_PROGRESS

SegmentSampleSpectral::SegmentSampleSpectral(TSampleSet *samples, TTriangleMesh *mesh) {

	mpSamples = samples;
	mpMesh = mesh;
}

SegmentSampleSpectral::~SegmentSampleSpectral() {
}

bool SegmentSampleSpectral::runSegmentation() {

	vector<vector<int>> sampleGraph;
	vector<bool> sampleFlag;
	if (!SegmentUtil::buildKNNGraph(*mpSamples, sampleGraph, sampleFlag)) return false;

	int numSamples = mpSamples->amount;

	// extract all edges
	set<pair<int, int>> neighborSet;
	for (int sampleID = 0; sampleID < numSamples; sampleID++) {
		if (!sampleFlag[sampleID]) continue;
		vec3 sampleN = mpSamples->normals[sampleID];
		for (int neighborID : sampleGraph[sampleID]) {
			if (!sampleFlag[neighborID]) continue;
			vec3 neighborN = mpSamples->normals[neighborID];
			if (cml::dot(sampleN, neighborN) < 0) continue;
			neighborSet.insert(make_pair(min(sampleID, neighborID), max(sampleID, neighborID)));
		}
	}

	// compute edge weights
	int numEdges = (int)neighborSet.size();
	vector<pair<int, int>> edgePairs(neighborSet.begin(), neighborSet.end());
	vector<Eigen::Triplet<double, int>> edgeTriplets(numEdges*2); // double edge
#pragma omp parallel for
	for (int edgeID = 0; edgeID < numEdges; edgeID++) {
		int id1 = edgePairs[edgeID].first;
		int id2 = edgePairs[edgeID].second;
		vec3 p1 = mpSamples->positions[id1];
		vec3 p2 = mpSamples->positions[id2];
		//vec3i tr1 = mpMesh->indices[mpSamples->indices[id1]];
		//vec3i tr2 = mpMesh->indices[mpSamples->indices[id2]];
		//vec3 n1 = cml::cross(mpMesh->positions[tr1[1]] - mpMesh->positions[tr1[0]], mpMesh->positions[tr1[2]] - mpMesh->positions[tr1[0]]);
		//vec3 n2 = cml::cross(mpMesh->positions[tr2[1]] - mpMesh->positions[tr2[0]], mpMesh->positions[tr2[2]] - mpMesh->positions[tr2[0]]);
		vec3 n1 = mpSamples->normals[id1];
		vec3 n2 = mpSamples->normals[id2];
		float a1 = cml::dot(cml::normalize(n1), cml::normalize(p2-p1));
		float a2 = cml::dot(cml::normalize(n2), cml::normalize(p1-p2));
		double weight;
		float eps = 0.01f;
		if (a1 <= eps && a2 <= eps) { // convex
			weight = 0.95;
		} else if (a1 > eps && a2 > eps) { // concave
			weight = 0.01;
		} else { // inconsistent neighbor normals
			weight = 0.5;
		}

		edgeTriplets[edgeID * 2] = Eigen::Triplet<double, int>(id1, id2, weight);
		edgeTriplets[edgeID * 2 + 1] = Eigen::Triplet<double, int>(id2, id1, weight);
	}

	//if (!visualizeGraph("Style/2.segment/debug/graph.ply", edgeTriplets)) return false;
	//if (!visualizeGraph("graph.ply", edgeTriplets)) return false;

	// add self edges
	for (int j = 0; j < numSamples; j++) {
		edgeTriplets.push_back(Eigen::Triplet<double, int>(j, j, 1.0));
	}

	// spectral clustering
	double eps = mpSamples->radius * 0.1;
	Eigen::SparseMatrix<double> adjMat(numSamples, numSamples);
	adjMat.setFromTriplets(edgeTriplets.begin(), edgeTriplets.end());
	Eigen::VectorXi clusters;
	if (!ClusterSparseSpectral::cluster(adjMat, clusters, eps)) return false;

	// export patches
	int numPatches = (int)clusters.maxCoeff()+1;
	mPatches.clear();
	mPatches.resize(numPatches, vector<int>(0));
	for (int sampleID = 0; sampleID < numSamples; sampleID++) {
		if (sampleFlag[sampleID]) {
			mPatches[clusters[sampleID]].push_back(sampleID);
		}
	}
#ifdef OUTPUT_PROGRESS
	cout << "Unpruned patches: " << numPatches << endl;
#endif

	// remove outliers
	vector<int> patchIDMap(numPatches, -1);
	if (mPatches.size()) {
		vector<vector<int>> tmpPatches;
		int newPatchID = 0;
		for (int patchID = 0; patchID < numPatches; patchID++) {
			if (mPatches[patchID].size() > 0) { // UNDONE: param min patch size
				tmpPatches.push_back(mPatches[patchID]);
				patchIDMap[patchID] = newPatchID;
				newPatchID++;
			}
		}
		mPatches.swap(tmpPatches);
		numPatches = newPatchID;
	}
#ifdef OUTPUT_PROGRESS
	cout << "Valid patches: " << numPatches << endl;
#endif

	// export patch graph
	set<pair<int, int>> patchNeighborSet;
	for (auto &it : neighborSet) {
		int patchID1 = patchIDMap[clusters[it.first]];
		int patchID2 = patchIDMap[clusters[it.second]];
		if (patchID1<0 || patchID2<0 || patchID1 == patchID2) continue;
		if (patchID1 > patchID2) swap(patchID1, patchID2);
		patchNeighborSet.insert(make_pair(patchID1, patchID2));
	}
	mPatchNeighbors.clear();
	mPatchNeighbors.resize(numPatches, vector<int>(0));
	for (auto &it : patchNeighborSet) {
		mPatchNeighbors[it.first].push_back(it.second);
		mPatchNeighbors[it.second].push_back(it.first);
	}

	return true;
}

bool SegmentSampleSpectral::exportSegmentation(vector<vector<int>> &outSegments, vector<vector<int>> &outGraph) {

	outSegments = mPatches;
	outGraph = mPatchNeighbors;

	return true;
}

bool SegmentSampleSpectral::visualizeSegmentation(string fileName) {

	// export segmented shape

	PlyExporter pe;
	int segmentID = 0;
	for (auto &patch : mPatches) {

		vec3i color;
		color = vec3i((segmentID % 6) * 50, (segmentID / 6 % 6) * 50, (segmentID / 36) * 50); // color coding for debug
		//color = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));

		vector<vec3> segmentPositions(0);
		vector<vec3> segmentNormals(0);
		for (int sampleID : patch) {
			segmentPositions.push_back(mpSamples->positions[sampleID]);
			segmentNormals.push_back(mpSamples->normals[sampleID]);
		}
		if (!pe.addPoint(&segmentPositions, &segmentNormals, cml::identity_4x4(), color)) return false;
		segmentID++;
	}

	if (!pe.output(fileName)) return false;

	return true;
}

bool SegmentSampleSpectral::visualizeGraph(string fileName, vector<Eigen::Triplet<double, int>> &graph) {

	// used for debug

	PlyExporter pe;

	float width = mpSamples->radius * 0.1f;

	vector<vec3i> vI; // indices of a rect
	vI.push_back(vec3i(0, 2, 1));
	vI.push_back(vec3i(1, 2, 3));

	for (auto &triplet : graph) {
		int id1 = triplet.row();
		int id2 = triplet.col();
		double weight = triplet.value();
		vec3 p1 = mpSamples->positions[id1];
		vec3 p2 = mpSamples->positions[id2];
		vec3 n1 = mpSamples->normals[id1];
		vec3 n2 = mpSamples->normals[id2];
		vec3 tgt = cml::normalize(cml::cross(n1, p2 - p1)) * width;
		vector<vec3> vP;
		vP.push_back(p1 - tgt);
		vP.push_back(p1 + tgt);
		vP.push_back(p2 - tgt);
		vP.push_back(p2 + tgt);

		vec3i color = vec3i(255, (int)(weight * 255), (int)(weight * 255));
		if (!pe.addMesh(&vI, &vP, 0, cml::identity_4x4(), color)) return false;
	}

	if (!pe.output(fileName)) return false;

	return true;
}