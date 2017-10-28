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

#include "FeatureTalSaliency.h"

#include <fstream>
#include <iostream>

#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"
#include "Feature/FeatureUtil.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

#define DEBUG_OUTPUT

FeatureTalSaliency::FeatureTalSaliency(TSampleSet *samples, vector<vector<double>> *saliencies) {

	mpSamples = samples;
	mpSaliencies = saliencies;

	// initializations
	SampleUtil::computeAABB(*samples, mBBMin, mBBMax);
	mLowRadius = samples->radius * 5.0; // UNDONE: param
	mHighRadius = max(mLowRadius*2, (mBBMax - mBBMin).length() * 0.1); // UNDONE: param

	if (!saliencies->empty()) {
		mSaliencyDLow.clear();
		mSaliencyALow.clear();
		mSaliencyDHigh.clear();
		for (auto &it : *saliencies) {
			if (it.size() != 3) break;
			mSaliencyDLow.push_back(it[0]);
			mSaliencyALow.push_back(it[1]);
			mSaliencyDHigh.push_back(it[2]);
		}
	}
}

FeatureTalSaliency::~FeatureTalSaliency() {
}

bool FeatureTalSaliency::calculateLowLevelSaliencies() {

	mNumPoints = (int)mLowFeatures.size();
	mNumBins = (int)mLowFeatures[0].size();

	if (!calculateDLow()) return false;
	//if (!extractDistinctPoints()) return false;

	return true;
}

bool FeatureTalSaliency::calculateOtherSaliencies() {

	if (!calculateALow()) return false;
	if (!calculateDHigh()) return false;

	return true;
}

bool FeatureTalSaliency::exportSaliencies() {

	mpSaliencies->resize(mNumPoints);
#pragma omp parallel for
	for (int pointID = 0; pointID < mNumPoints; pointID++) {
		auto &saliency = (*mpSaliencies)[pointID];
		saliency.clear();
		saliency.push_back(mSaliencyDLow[pointID]);
		saliency.push_back(mSaliencyALow[pointID]);
		saliency.push_back(mSaliencyDHigh[pointID]);
	}

	return true;
}

bool FeatureTalSaliency::calculateDLow() {

#ifdef DEBUG_OUTPUT
	cout << "Computing DLow..." << endl;
#endif

	int kNeighbors = (int)(mNumPoints * 0.01); // UNDONE: param: percentage of k neighbors
	float bbLength = (mBBMax - mBBMin).length(); // mesh diagonal length

	// build kd tree with flann

	flann::Matrix<double> treeData(new double[mNumPoints*mNumBins], mNumPoints, mNumBins);
	for (int r = 0; r<mNumPoints; r++) {
		for (int c = 0; c<mNumBins; c++) {
			treeData[r][c] = mLowFeatures[r][c];
		}
	}
	flann::Index<flann::ChiSquareDistance<double>> tree(treeData, flann::KDTreeIndexParams(1));
	tree.buildIndex();

	// search neighbors in histogram space

	flann::SearchParams params;
	params.checks = -1;
	params.cores = 0;

	mSaliencyDLow.resize(mNumPoints);
#pragma omp parallel for
	for (int pointID = 0; pointID < mNumPoints; pointID++) {

		flann::Matrix<double> queryData(new double[mNumBins], 1, mNumBins);
		flann::Matrix<int> queryIdx(new int[kNeighbors], 1, kNeighbors);
		flann::Matrix<double> queryDist(new double[kNeighbors], 1, kNeighbors);
		for (int i = 0; i<mNumBins; i++) queryData[0][i] = mLowFeatures[pointID][i];

		tree.knnSearch(queryData, queryIdx, queryDist, kNeighbors, params);

		double totalDist = 0;
		int totalCount = 0;
		for (int i = 0; i < kNeighbors; i++) {
			int neighborID = queryIdx[0][i];
			if (neighborID == pointID) continue;
			double histDist = queryDist[0][i];
			double pointDist = (mpSamples->positions[pointID] - mpSamples->positions[neighborID]).length() / bbLength;
			double dist = histDist / (1 + pointDist);
			totalDist += dist;
			totalCount++;
		}
		double distinctness = 1 - exp(-totalDist / totalCount);
		mSaliencyDLow[pointID] = distinctness;

		delete[] queryData.ptr();
		delete[] queryIdx.ptr();
		delete[] queryDist.ptr();
	}

	return true;
}

bool FeatureTalSaliency::extractDistinctPoints() {

	const double distinctPointFraction = 0.1; // UNDONE: param fraction of points used to compute high level descriptor

	// get min distinctness for distinct points
	double distinctMinDLow = 0;
	{
		vector<double> vecDLow(mSaliencyDLow);
		int n = (int)(mNumPoints*(1 - distinctPointFraction));
		nth_element(vecDLow.begin(), vecDLow.begin() + n, vecDLow.end());
		distinctMinDLow = vecDLow[n];
	}

	// extract sample set (put original sample IDs into indices)
	mDistinctPoints.amount = (int)(mNumPoints*distinctPointFraction);
	mDistinctPoints.indices.clear();
	mDistinctPoints.positions.clear();
	mDistinctPoints.normals.clear();
	mDistinctPoints.indices.reserve(mDistinctPoints.amount);
	mDistinctPoints.positions.reserve(mDistinctPoints.amount);
	mDistinctPoints.normals.reserve(mDistinctPoints.amount);
	for (int pointID = 0; pointID < mNumPoints; pointID++) {
		if (mSaliencyDLow[pointID] >= distinctMinDLow) {
			mDistinctPoints.indices.push_back(pointID);
			mDistinctPoints.positions.push_back(mpSamples->positions[pointID]);
			mDistinctPoints.normals.push_back(mpSamples->normals[pointID]);
		}
	}
	mDistinctPoints.amount = (int)mDistinctPoints.positions.size();

	//if (!SampleUtil::saveSample("Style/2.feature/saliency/distinct.ply", mDistinctPoints)) return false;

	return true;
}

bool FeatureTalSaliency::calculateALow() {

#ifdef DEBUG_OUTPUT
	cout << "Computing ALow..." << endl;
#endif

	const double focusPointFraction = 0.2; // UNDONE: param fraction of focus points with highest distinction
	const double associationDeviation = 3.0; // UNDONE: param Gaussian sigma for contribution from associated point
	double associationWeight = 1 / (2 * cml::sqr(mpSamples->radius * associationDeviation));

	// get min distinctness for focus points

	double focusMinDLow = 0;
	{
		vector<double> vecDLow(mSaliencyDLow);
		int n = (int)(mNumPoints*(1-focusPointFraction));
		nth_element(vecDLow.begin(), vecDLow.begin() + n, vecDLow.end());
		focusMinDLow = vecDLow[n];
	}

	// build KD tree for focus points

	SKDTreeData treeData;
	treeData.reserve((int)(mNumPoints*focusPointFraction));
	for (int pointID = 0; pointID < mNumPoints; pointID++) {
		if (mSaliencyDLow[pointID] >= focusMinDLow) {
			vec3 &v = mpSamples->positions[pointID];
			treeData.push_back(SKDT::NamedPoint(v[0], v[1], v[2], pointID));
		}
	}
	SKDTree tree(treeData.begin(), treeData.end());

	// compute point association

	mSaliencyALow.resize(mNumPoints);
#pragma omp parallel for
	for (int pointID = 0; pointID < mNumPoints; pointID++) {

		vec3 nowPoint = mpSamples->positions[pointID];
		double nowALow = 0;

		SKDT::NamedPoint queryPoint(nowPoint[0], nowPoint[1], nowPoint[2]);
		Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1);
		tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
		if (!queryResult.isEmpty()) {
			int focusID = (int)tree.getElements()[queryResult[0].getIndex()].id;
			vec3 focusPoint = mpSamples->positions[focusID];
			nowALow = mSaliencyDLow[focusID] * exp(-(nowPoint - focusPoint).length_squared() * associationWeight);
		}

		mSaliencyALow[pointID] = nowALow;
	}

	return true;
}

bool FeatureTalSaliency::calculateDHigh() {

#ifdef DEBUG_OUTPUT
	cout << "Computing DHigh..." << endl;
#endif

	int kNeighbors = (int)(mNumPoints * 0.1); // UNDONE: param: same with the one in DLow computation
	float bbLength = (mBBMax - mBBMin).length(); // mesh diagonal length

	// build kd tree with flann
	flann::Matrix<double> treeData(new double[mNumPoints*mNumBins], mNumPoints, mNumBins);
	for (int r = 0; r<mNumPoints; r++) {
		for (int c = 0; c<mNumBins; c++) {
			treeData[r][c] = mHighFeatures[r][c];
		}
	}
	flann::Index<flann::ChiSquareDistance<double>> tree(treeData, flann::KDTreeIndexParams(1));
	tree.buildIndex();

	// search neighbors in histogram space

	flann::SearchParams params;
	params.checks = -1;
	params.cores = 0;

	mSaliencyDHigh.resize(mNumPoints);
#pragma omp parallel for
	for (int pointID = 0; pointID < mNumPoints; pointID++) {

		flann::Matrix<double> queryData(new double[mNumBins], 1, mNumBins);
		flann::Matrix<int> queryIdx(new int[kNeighbors], 1, kNeighbors);
		flann::Matrix<double> queryDist(new double[kNeighbors], 1, kNeighbors);
		for (int i = 0; i<mNumBins; i++) queryData[0][i] = mHighFeatures[pointID][i];

		tree.knnSearch(queryData, queryIdx, queryDist, kNeighbors, params);

		double totalDist = 0;
		int totalCount = 0;
		for (int i = 0; i < kNeighbors; i++) {
			int neighborID = queryIdx[0][i];
			if (neighborID == pointID) continue;
			double histDist = queryDist[0][i];
			double pointDist = (mpSamples->positions[pointID] - mpSamples->positions[neighborID]).length() / bbLength;
			double dist = histDist * log(1 + pointDist);

			totalDist += dist;
			totalCount++;
		}
		double distinctness = 1 - exp(-totalDist / totalCount);
		mSaliencyDHigh[pointID] = distinctness;

		delete[] queryData.ptr();
		delete[] queryIdx.ptr();
		delete[] queryDist.ptr();
	}

	return true;
}

bool FeatureTalSaliency::visualize(string fileName) {

	int numPoints = mpSamples->amount;

	double maxDLow = mSaliencyDLow[0];
	double minDLow = maxDLow;
	double maxALow = mSaliencyALow[0];
	double minALow = maxALow;
	double maxDHigh = mSaliencyDHigh[0];
	double minDHigh = maxDHigh;
	for (int pointID = 0; pointID < numPoints; pointID++) {
		maxDLow = max(maxDLow, mSaliencyDLow[pointID]);
		minDLow = min(minDLow, mSaliencyDLow[pointID]);
		maxALow = max(maxALow, mSaliencyALow[pointID]);
		minALow = min(minALow, mSaliencyALow[pointID]);
		maxDHigh = max(maxDHigh, mSaliencyDHigh[pointID]);
		minDHigh = min(minDHigh, mSaliencyDHigh[pointID]);
	}

	//cout << "Max Saliency: " << maxDLow << ", " << maxALow << ", " << maxDHigh << endl;
	//cout << "Min Saliency: " << minDLow << ", " << minALow << ", " << minDHigh << endl;

	vector<vec3i> vColorsDLow;
	vector<vec3i> vColorsALow;
	vector<vec3i> vColorsDHigh;
	for (int pointID = 0; pointID < numPoints; pointID++) {
		double vDLow = (mSaliencyDLow[pointID] - minDLow) / (maxDLow - minDLow);
		double vALow = (mSaliencyALow[pointID] - minALow) / (maxALow - minALow);
		double vDHigh = (mSaliencyDHigh[pointID] - minDHigh) / (maxDHigh - minDHigh);
		vColorsDLow.push_back(FeatureUtil::colorMapping(vDLow));
		vColorsALow.push_back(FeatureUtil::colorMapping(vALow));
		vColorsDHigh.push_back(FeatureUtil::colorMapping(vDHigh));
	}
	
	vec3 offset((mBBMax[0] - mBBMin[0])*1.2f, 0.0f, 0.0f);

	PlyExporter pe;
	if (!pe.addPoint(&mpSamples->positions, &mpSamples->normals, &vColorsDLow, -offset)) return false;
	if (!pe.addPoint(&mpSamples->positions, &mpSamples->normals, &vColorsALow)) return false;
	if (!pe.addPoint(&mpSamples->positions, &mpSamples->normals, &vColorsDHigh, offset)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool FeatureTalSaliency::saveFeature(string fileName) {

	ofstream outFile(fileName, ios::binary);
	if (!outFile.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}

	for (auto &it : *mpSaliencies) {
		for (auto value : it) {
			outFile.write((const char*)(&value), sizeof(value));
		}
	}
	outFile.close();

	return true;
}

bool FeatureTalSaliency::loadFeature(string fileName, vector<vector<double>> &feature) {

	const int numSaliencyMetrics = 3; // fixed

	ifstream inFile(fileName, ios::binary | ios::ate);
	if (!inFile.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	std::streampos fileSize = inFile.tellg();
	int n = (int)(fileSize / (numSaliencyMetrics * sizeof(feature[0][0])));
	feature.resize(n);

	inFile.seekg(0, ios::beg);
	for (int r = 0; r < n; r++) {
		feature[r].resize(numSaliencyMetrics);
		for (int c = 0; c < numSaliencyMetrics; c++) {
			inFile.read((char*)(&feature[r][c]), sizeof(feature[0][0]));
		}
	}
	inFile.close();

	return true;
}