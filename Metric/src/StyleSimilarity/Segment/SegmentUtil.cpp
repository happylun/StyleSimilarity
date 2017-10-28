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

#include "SegmentUtil.h"

#include <fstream>
#include <set>

#include <Library/CMLHelper.h>

#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

bool SegmentUtil::saveSegmentationData(string fileName, vector<vector<int>> &segmentData) {

	ofstream outFile(fileName);

	outFile << segmentData.size() << endl;
	for(auto &segment : segmentData) {
		outFile << segment.size() << " ";
		for(auto &sampleID : segment) {
			outFile << sampleID << " ";
		}
		outFile << endl;
	}

	outFile.close();

	return true;
}

bool SegmentUtil::loadSegmentationData(string fileName, vector<vector<int>> &segmentData) {

	ifstream inFile(fileName);
	if(!inFile.is_open()) {
		cout << "Error: cannot load segmentation file " << fileName << endl;
		return false;
	}

	int numSegments = 0;
	inFile >> numSegments;
	segmentData.clear();
	segmentData.reserve(numSegments);

	for (int segID = 0; segID < numSegments; segID++) {

		int numSamples;
		inFile >> numSamples;
		if(!inFile.good()) break;

		vector<int> segment;
		segment.resize(numSamples);
		for(int i=0; i<numSamples; i++) {
			int sampleID;
			inFile >> sampleID;
			segment[i] = sampleID;
		}
		if (segment.size() >= 20) segmentData.push_back(segment);
	}

	inFile.close();

	return true;
}

bool SegmentUtil::savePatchData(string fileName, vector<vector<int>> &patchData, vector<vector<int>> &patchGraph) {

	if(patchData.size() != patchGraph.size()) {
		cout << "Error: inconsistent number of patches" << endl;
		return false;
	}

	ofstream outFile(fileName);

	outFile << patchData.size() << endl;

	for(auto &patch : patchData) {
		outFile << patch.size() << " ";
		for(auto &sampleID : patch) {
			outFile << sampleID << " ";
		}
		outFile << endl;
	}

	for(auto &node : patchGraph) {
		outFile << node.size() << " ";
		for(auto &neighborID : node) {
			outFile << neighborID << " ";
		}
		outFile << endl;
	}

	outFile.close();

	return true;
}

bool SegmentUtil::loadPatchData(string fileName, vector<vector<int>> &patchData, vector<vector<int>> &patchGraph) {

	ifstream inFile(fileName);
	if(!inFile.is_open()) {
		cout << "Error: cannot load patch file " << fileName << endl;
		return false;
	}

	int numPatches = 0;
	inFile >> numPatches;

	patchData.clear();
	patchData.resize(numPatches);
	for(auto &patch : patchData) {

		int numSamples;
		inFile >> numSamples;
		if(!inFile.good()) break;

		patch.clear();
		patch.resize(numSamples);
		for(int i=0; i<numSamples; i++) {
			int sampleID;
			inFile >> sampleID;
			patch[i] = sampleID;
		}
	}

	patchGraph.clear();
	patchGraph.resize(numPatches);
	for(auto &node : patchGraph) {

		int numNeighbors;
		inFile >> numNeighbors;
		if(!inFile.good()) break;

		node.clear();
		node.resize(numNeighbors);
		for(int i=0; i<numNeighbors; i++) {
			int neighborID;
			inFile >> neighborID;
			node[i] = neighborID;
		}
	}

	inFile.close();

	return true;
}

bool SegmentUtil::visualizeSegmentation(string fileName, TSampleSet &samples, vector<vector<int>> &segments, bool indexedColor) {

	vec3 bbMin, bbMax;
	if (!SampleUtil::computeAABB(samples, bbMin, bbMax)) return false;
	float horizontalSpacing = (bbMax[0] - bbMin[0]) * 1.2f;

	PlyExporter pe;
	int segmentID = 0;
	int groupID = 0;
	set<int> visitedPointSet;
	for (auto &segment : segments) {

		for (int pointID : segment) {
			if (visitedPointSet.find(pointID) != visitedPointSet.end()) {
				visitedPointSet.clear();
				groupID++;
			}
		}
		visitedPointSet.insert(segment.begin(), segment.end());
		vec3 offset(horizontalSpacing * groupID, 0.0f, 0.0f);

		vec3i color = colorMapping(segmentID);
		if (indexedColor) {
			color = vec3i((segmentID % 6) * 50, (segmentID / 6 % 6) * 50, (segmentID / 36) * 50); // color coding for debug
		}

		TPointSet ps;
		if (!extractPointSet(samples, segment, ps)) return false;
		if (!pe.addPoint(&ps.positions, &ps.normals, offset, color)) return false;
		segmentID++;
	}

	if (!pe.output(fileName)) return false;

	return true;
}

bool SegmentUtil::visualizeSegmentationGraph(string fileName, TSampleSet &samples, vector<vector<int>> &segments, vector<vector<int>> &graph) {

	int numSegments = (int)segments.size();
	vector<vec3> segmentCenter(numSegments);
	for (int segmentID = 0; segmentID<numSegments; segmentID++) {
		vec3 center(0.0f, 0.0f, 0.0f);
		for (int pointID : segments[segmentID]) {
			center += samples.positions[pointID];
		}
		segmentCenter[segmentID] = center / (int)segments[segmentID].size();
	}

	vector<vec3> edgeList;
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		for (int neighborID : graph[segmentID]) {
			edgeList.push_back(segmentCenter[segmentID]);
			edgeList.push_back(segmentCenter[neighborID]);
		}
	}

	PlyExporter pe;
	if (!pe.addLine(&edgeList)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool SegmentUtil::buildKNNGraph(
	TSampleSet &inSamples,
	vector<vector<int>> &outGraph,
	vector<bool> &outFlag)
{
	SKDTree tree;
	SKDTreeData treeData;
	if (!SampleUtil::buildKdTree(inSamples.positions, tree, treeData)) return false;

	vector<set<int>> sampleCloseNeighbors(inSamples.amount);

	// build graph
	const int numNeighbors = 7; // used for build graph
#pragma omp parallel for
	for (int sampleID = 0; sampleID < inSamples.amount; sampleID++) {
		vec3 sampleP = inSamples.positions[sampleID];
		SKDT::NamedPoint queryPoint(sampleP[0], sampleP[1], sampleP[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(20);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult, inSamples.radius*1.5);
		if (queryResult.size() < numNeighbors) {
			queryResult.clear();
			queryResult = Thea::BoundedSortedArray<SKDTree::Neighbor>(numNeighbors);
			tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
		}
		for (int id = 0; id < queryResult.size(); id++) {
			int neighborID = (int)tree.getElements()[queryResult[id].getIndex()].id;
			if (sampleID == neighborID) continue;
			sampleCloseNeighbors[sampleID].insert(neighborID);
		}
	}

	// detect outliers
	outFlag.assign(inSamples.amount, true);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < inSamples.amount; sampleID++) {
		vec3 samplePos = inSamples.positions[sampleID];
		SKDT::NamedPoint queryPoint(samplePos[0], samplePos[1], samplePos[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(2);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
		if (queryResult.size() > 1) {
			int neighborID = (int)tree.getElements()[queryResult[1].getIndex()].id;
			auto &topSet = sampleCloseNeighbors[neighborID];
			if (topSet.find(sampleID) == topSet.end()) outFlag[sampleID] = false;
		}
		else {
			outFlag[sampleID] = false;
		}
	}

	// export graph
	outGraph.resize(inSamples.amount);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < inSamples.amount; sampleID++) {
		if (outFlag[sampleID]) {
			auto &nbSet = sampleCloseNeighbors[sampleID];
			outGraph[sampleID].assign(nbSet.begin(), nbSet.end());
		} else {
			outGraph[sampleID].clear();
		}
	}

	return true;
}

bool SegmentUtil::extractFinestSegmentation(
	TSampleSet &inSamples,
	vector<vector<int>> &inSegments,
	vector<vector<int>> &outSegments,
	vector<vector<int>> &outSegmentsGraph,
	int level)
{
	// Extract finest resolution segmentation and the graph between segments from existing segmentation data.
	// This piece of code is a bit redundant. I can extract this within the segmentation code.
	// But I don't want to re-generating existing segmentation data (and all followings...).
	// That is why I wrote this...

	///////////////// extract finest segmentation /////////////////

	// find non-overlapping segment set

	map<int, int> segmentMap; // key: pointID; value: segmentID
	set<int> visitedPointSet;
	visitedPointSet.clear();
	int groupID = 0;
	int startID = 0;
	int finishID = -1;
	for (int segmentID = 0; segmentID < (int)inSegments.size(); segmentID++) {
		auto &segment = inSegments[segmentID];
		for (int pointID : segment) {
			if (visitedPointSet.find(pointID) != visitedPointSet.end()) {
				groupID++;
				if (groupID > level) {
					finishID = segmentID;
					break;
				}
				startID = segmentID;
				visitedPointSet.clear();
				segmentMap.clear();
			}
		}
		if (groupID > level) break;
		
		visitedPointSet.insert(segment.begin(), segment.end());		
		for (int pointID : segment) {
			segmentMap[pointID] = segmentID - startID;
		}
	}

	// export segments
	if (finishID < 0) finishID = (int)inSegments.size();
	outSegments.assign(inSegments.begin() + startID, inSegments.begin() + finishID);
	int numSegments = finishID - startID;

	///////////////// extract graph /////////////////

	vector<vector<int>> sampleGraph;
	vector<bool> sampleFlag;
	if (!buildKNNGraph(inSamples, sampleGraph, sampleFlag)) return false;

	// extract all edges
	vector<set<int>> segmentNeighborSet(numSegments);
	for (int sampleID = 0; sampleID < inSamples.amount; sampleID++) {
		if (!sampleFlag[sampleID]) continue;
		auto sampleIter = segmentMap.find(sampleID);
		if (sampleIter == segmentMap.end()) continue;
		int sampleSeg = sampleIter->second;

		vec3 sampleN = inSamples.normals[sampleID];
		for (int neighborID : sampleGraph[sampleID]) {
			if (!sampleFlag[neighborID]) continue;
			vec3 neighborN = inSamples.normals[neighborID];
			if (cml::dot(sampleN, neighborN) < 0) continue;

			auto neighborIter = segmentMap.find(neighborID);
			if (neighborIter == segmentMap.end()) continue;
			int neighborSeg = neighborIter->second;

			segmentNeighborSet[sampleSeg].insert(neighborSeg);
			segmentNeighborSet[neighborSeg].insert(sampleSeg);
		}
	}

	// export edges
	outSegmentsGraph.resize(numSegments);
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		auto &ns = segmentNeighborSet[segmentID];
		ns.erase(segmentID);
		outSegmentsGraph[segmentID].assign(ns.begin(), ns.end());
	}

	return true;
}

bool SegmentUtil::extractPointSet(TPointSet &inSamples, vector<int> &inSegment, TPointSet &outPointSet) {

	int numSamples = (int)inSegment.size();
	outPointSet.amount = numSamples;
	outPointSet.positions.resize(numSamples);
	outPointSet.normals.resize(numSamples);

	for (int i = 0; i<numSamples; i++) {
		outPointSet.positions[i] = inSamples.positions[inSegment[i]];
		outPointSet.normals[i] = inSamples.normals[inSegment[i]];
	}

	return true;
}

bool SegmentUtil::extractPointSet(TPointSet &inSamples, vector<vector<int>> &inSegments, vector<TPointSet> &outPointSets) {

	outPointSets.resize(inSegments.size());

	for (int segID = 0; segID < (int)inSegments.size(); segID++) {
		auto &segment = inSegments[segID];
		TPointSet &points = outPointSets[segID];
		if (!extractPointSet(inSamples, segment, points)) return false;
	}

	return true;
}

bool SegmentUtil::extractSampleSet(TSampleSet &inSamples, vector<int> &inSegment, TSampleSet &outSampleSet) {

	if (!extractPointSet(inSamples, inSegment, outSampleSet)) return false;

	outSampleSet.radius = inSamples.radius;
	outSampleSet.indices.clear();
	outSampleSet.indices.reserve(inSegment.size());
	for (int idx : inSegment) {
		outSampleSet.indices.push_back(inSamples.indices[idx]);
	}

	return true;
}

bool SegmentUtil::extractSampleSet(TSampleSet &inSamples, vector<vector<int>> &inSegments, vector<TSampleSet> &outSampleSets) {

	outSampleSets.resize(inSegments.size());

	for (int segID = 0; segID < (int)inSegments.size(); segID++) {
		auto &segment = inSegments[segID];
		TSampleSet &samples = outSampleSets[segID];
		if (!extractSampleSet(inSamples, segment, samples)) return false;
	}

	return true;
}

bool SegmentUtil::calculateNormalMap(vector<vec3> &inNormal, vector<double> &outNormalMap) {

	const int GRID_SIZE = 10; // UNDONE: param
	const int NUM_BINS = GRID_SIZE*GRID_SIZE*2;

	static vector<vec3> binCenterDir(0);
	if (binCenterDir.empty()) {
		binCenterDir.resize(NUM_BINS);
		for (int y = 0; y < GRID_SIZE; y++) {
			float phi = (y + 0.5f) * cml::constantsf::pi() / (GRID_SIZE);
			for (int x = 0; x < GRID_SIZE * 2; x++) {
				float theta = (x + 0.5f) * cml::constantsf::two_pi() / (GRID_SIZE * 2);
				vec3 dir;
				cml::spherical_to_cartesian(1.0f, theta, phi, 1, cml::colatitude, dir);
				int index = y*GRID_SIZE * 2 + x;
				binCenterDir[index] = cml::normalize(dir);
			}
		}
	}

	double kappaN = 10.0; // UNDONE: param

	outNormalMap.resize(NUM_BINS, 0);
	int totalCount = 0;
	for(vec3 n : inNormal) {
		vector<double> smoothHist(NUM_BINS);
		double totalWeight = 0;
		for(int index=0; index<NUM_BINS; index++) {
			double cosN = (double) cml::dot(binCenterDir[index], n);
			double weight = exp( (cosN-1.0) * kappaN ); // kappa=10 => 0 deg: 1.000   30 deg: 0.262   60 deg: 0.007
			if(weight < 1e-3) weight=0;
			smoothHist[index] = weight;
			totalWeight += weight;
		}
		for(double &weight : smoothHist) weight /= totalWeight;
		for(int index=0; index<NUM_BINS; index++) {
			outNormalMap[index] += smoothHist[index];
		}
		totalCount++;
	}
	if(totalCount == 0) return false;
	for(auto &v : outNormalMap) v /= totalCount;

	return true;
}

bool SegmentUtil::compareNormalMap(vector<double> &inSourceNormalMap, vector<double> &inTargetNormalMap, double &outSimilarity) {

	if( inSourceNormalMap.size() != inTargetNormalMap.size() ) return false;
	int numBins = (int)inSourceNormalMap.size();

	// calculate histogram overlap distance
	double overlap = 0;
	for(int j=0; j<numBins; j++) {
		overlap += min(inSourceNormalMap[j], inTargetNormalMap[j]);
	}

	outSimilarity = overlap;

	return true;
}


bool SegmentUtil::labelSampleSet(TSampleSet &inSamples, vector<TTriangleMesh> &inParts, vector<vector<int>> &outSegments) {

	int numParts = (int)inParts.size();
	int numSamples = inSamples.amount;

	// build KD tree

	TKDTreeData treeData;
	TKDTree tree;

	vector<int> faceIdexOffset;
	int totalFaces = 0;
	for (int partID = 0; partID < numParts; partID++) {
		TTriangleMesh &part = inParts[partID];
		faceIdexOffset.push_back(totalFaces);
		for (int faceID = 0; faceID < (int)part.indices.size(); faceID++) {
			vec3i idx = part.indices[faceID];
			G3D::Vector3 v0(part.positions[idx[0]].data());
			G3D::Vector3 v1(part.positions[idx[1]].data());
			G3D::Vector3 v2(part.positions[idx[2]].data());

			TKDTreeElement tri(TKDT::NamedTriangle(v0, v1, v2, totalFaces));
			treeData.push_back(tri);

			totalFaces++;
		}
	}
	tree.init(treeData.begin(), treeData.end());

	// find label of nearest face

	vector<int> labels(numSamples, -1);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < numSamples; sampleID++) {
		vec3 sampleP = inSamples.positions[sampleID];
		SKDT::NamedPoint queryPoint(sampleP[0], sampleP[1], sampleP[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
		if (!queryResult.isEmpty()) {
			int neighborID = (int)queryResult[0].getIndex();
			for (int k = numParts - 1; k >= 0; k--) {
				if (neighborID >= faceIdexOffset[k]) {
					labels[sampleID] = k;
					break;
				}
			}
		}
	}

	outSegments.assign(numParts, vector<int>(0));
	for (int sampleID = 0; sampleID < numSamples; sampleID++) {
		if (labels[sampleID] >= 0) {
			outSegments[labels[sampleID]].push_back(sampleID);
		}
	}

	return true;
}

vec3i SegmentUtil::colorMapping(int index) {

	static vector<vec3i> colorMap(0);
	if (colorMap.size() == 0) {
		// random permutation of HSV color map
		colorMap.push_back(vec3i(255, 0, 0));
		colorMap.push_back(vec3i(0, 128, 128));
		colorMap.push_back(vec3i(0, 255, 0));
		colorMap.push_back(vec3i(128, 0, 128));
		colorMap.push_back(vec3i(255, 255, 0));
		colorMap.push_back(vec3i(0, 0, 255));
		colorMap.push_back(vec3i(0, 128, 0));
		colorMap.push_back(vec3i(255, 0, 255));		
		colorMap.push_back(vec3i(0, 255, 255));
		colorMap.push_back(vec3i(0, 0, 128));
		colorMap.push_back(vec3i(128, 0, 0));
		colorMap.push_back(vec3i(192, 192, 192));
		colorMap.push_back(vec3i(128, 128, 0));
		colorMap.push_back(vec3i(111, 0, 255));
		colorMap.push_back(vec3i(0, 120, 255));
		colorMap.push_back(vec3i(0, 255, 184));
		colorMap.push_back(vec3i(255, 0, 141));
		colorMap.push_back(vec3i(255, 103, 0));
		colorMap.push_back(vec3i(0, 107, 255));
		colorMap.push_back(vec3i(255, 231, 0));
		colorMap.push_back(vec3i(60, 0, 255));
		colorMap.push_back(vec3i(21, 255, 0));
		colorMap.push_back(vec3i(21, 0, 255));
		colorMap.push_back(vec3i(124, 255, 0));
		colorMap.push_back(vec3i(73, 255, 0));
		colorMap.push_back(vec3i(0, 255, 107));
		colorMap.push_back(vec3i(255, 0, 77));
		colorMap.push_back(vec3i(137, 255, 0));
		colorMap.push_back(vec3i(0, 255, 43));
		colorMap.push_back(vec3i(255, 0, 26));
		colorMap.push_back(vec3i(0, 255, 159));
		colorMap.push_back(vec3i(0, 236, 255));
		colorMap.push_back(vec3i(0, 210, 255));
		colorMap.push_back(vec3i(34, 0, 255));
		colorMap.push_back(vec3i(201, 255, 0));
		colorMap.push_back(vec3i(9, 255, 0));
		colorMap.push_back(vec3i(86, 0, 255));
		colorMap.push_back(vec3i(255, 13, 0));
		colorMap.push_back(vec3i(201, 0, 255));
		colorMap.push_back(vec3i(240, 0, 255));
		colorMap.push_back(vec3i(255, 0, 51));
		colorMap.push_back(vec3i(0, 255, 197));
		colorMap.push_back(vec3i(0, 255, 56));
		colorMap.push_back(vec3i(189, 255, 0));
		colorMap.push_back(vec3i(176, 0, 255));
		colorMap.push_back(vec3i(255, 90, 0));
		colorMap.push_back(vec3i(0, 133, 255));
		colorMap.push_back(vec3i(255, 0, 90));
		colorMap.push_back(vec3i(34, 255, 0));
		colorMap.push_back(vec3i(255, 0, 154));
		colorMap.push_back(vec3i(214, 255, 0));
		colorMap.push_back(vec3i(189, 0, 255));
		colorMap.push_back(vec3i(0, 30, 255));
		colorMap.push_back(vec3i(99, 0, 255));
		colorMap.push_back(vec3i(255, 64, 0));
		colorMap.push_back(vec3i(255, 0, 206));
		colorMap.push_back(vec3i(255, 0, 39));
		colorMap.push_back(vec3i(240, 255, 0));
		colorMap.push_back(vec3i(255, 0, 0));
		colorMap.push_back(vec3i(0, 4, 255));
		colorMap.push_back(vec3i(60, 255, 0));
		colorMap.push_back(vec3i(0, 69, 255));
		colorMap.push_back(vec3i(255, 116, 0));
		colorMap.push_back(vec3i(255, 0, 13));
		colorMap.push_back(vec3i(86, 255, 0));
		colorMap.push_back(vec3i(253, 0, 255));
		colorMap.push_back(vec3i(0, 223, 255));
		colorMap.push_back(vec3i(255, 180, 0));
		colorMap.push_back(vec3i(255, 77, 0));
		colorMap.push_back(vec3i(255, 0, 219));
		colorMap.push_back(vec3i(99, 255, 0));
		colorMap.push_back(vec3i(0, 255, 171));
		colorMap.push_back(vec3i(0, 255, 236));
		colorMap.push_back(vec3i(0, 255, 223));
		colorMap.push_back(vec3i(255, 39, 0));
		colorMap.push_back(vec3i(0, 255, 133));
		colorMap.push_back(vec3i(227, 0, 255));
		colorMap.push_back(vec3i(0, 255, 69));
		colorMap.push_back(vec3i(0, 159, 255));
		colorMap.push_back(vec3i(255, 219, 0));
		colorMap.push_back(vec3i(255, 0, 231));
		colorMap.push_back(vec3i(73, 0, 255));
		colorMap.push_back(vec3i(0, 184, 255));
		colorMap.push_back(vec3i(0, 255, 81));
		colorMap.push_back(vec3i(0, 146, 255));
		colorMap.push_back(vec3i(255, 0, 129));
		colorMap.push_back(vec3i(255, 0, 193));
		colorMap.push_back(vec3i(0, 94, 255));
		colorMap.push_back(vec3i(0, 56, 255));
		colorMap.push_back(vec3i(0, 171, 255));
		colorMap.push_back(vec3i(111, 255, 0));
		colorMap.push_back(vec3i(0, 255, 210));
		colorMap.push_back(vec3i(255, 167, 0));
		colorMap.push_back(vec3i(0, 255, 30));
		colorMap.push_back(vec3i(0, 255, 249));
		colorMap.push_back(vec3i(137, 0, 255));
		colorMap.push_back(vec3i(255, 206, 0));
		colorMap.push_back(vec3i(255, 193, 0));
		colorMap.push_back(vec3i(163, 255, 0));
		colorMap.push_back(vec3i(0, 255, 146));
		colorMap.push_back(vec3i(163, 0, 255));
		colorMap.push_back(vec3i(255, 26, 0));
		colorMap.push_back(vec3i(0, 255, 17));
		colorMap.push_back(vec3i(0, 197, 255));
		colorMap.push_back(vec3i(150, 255, 0));
		colorMap.push_back(vec3i(0, 43, 255));
		colorMap.push_back(vec3i(255, 129, 0));
		colorMap.push_back(vec3i(255, 0, 64));
		colorMap.push_back(vec3i(255, 141, 0));
		colorMap.push_back(vec3i(47, 0, 255));
		colorMap.push_back(vec3i(255, 154, 0));
		colorMap.push_back(vec3i(176, 255, 0));
	}

	if (index + 1 < (int)colorMap.size()) return colorMap[index];

	return vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
}