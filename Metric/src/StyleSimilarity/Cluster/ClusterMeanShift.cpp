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

#include "ClusterMeanShift.h"

#include <iostream>
#include <algorithm>

#include "Library/CMLHelper.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

ClusterMeanShift::ClusterMeanShift(vector<vector<double>> &points, vector<double> &weights) {

	clearUp();

	mpDataPoints = &points;
	mpDataWeights = &weights;

	mNumPoints = (int)points.size();
	if( !mNumPoints ) {
		cout << "Error: empty data" << endl;
	}
	if( (int)weights.size() != mNumPoints ) {
		cout << "Error: data size incorrect" << endl;
	}

	mDimensions = (int)points[0].size();
	if( !mDimensions ) {
		cout << "Error: data dimension error" << endl;
	}

	// build kd tree with flann
	mpKdTreeData = new flann::Matrix<double>(new double[mNumPoints*mDimensions], mNumPoints, mDimensions);
	for(int r=0; r<mNumPoints; r++) {
		for(int c=0; c<mDimensions; c++) {
			(*mpKdTreeData)[r][c] = points[r][c];
		}
	}
	mpKdTree = new flann::Index<flann::L2<double>>(*mpKdTreeData, flann::KDTreeIndexParams(1));
	mpKdTree->buildIndex();

	// calculate sufficient weight for determining adaptive bandwidth
	mSufficientWeight = StyleSimilarityConfig::mCluster_BandwidthCumulatedWeight;

	mFixedBandwidth = -1; // use adaptive bandwidth by default
}

ClusterMeanShift::~ClusterMeanShift() {
	clearUp();
}

bool ClusterMeanShift::setBandwidth(double bandwidth) {

	mFixedBandwidth = bandwidth;

	return true;
}

bool ClusterMeanShift::runClustering(vector<int> &modes, vector<int> &modeIndices) {

	if (mNumPoints <= 20) {
		// too few points...
		// use each point as a cluster
		modes.resize(mNumPoints);
		for (int pointID = 0; pointID < mNumPoints; pointID++) {
			modes[pointID] = pointID;
		}
		return true;
	}

	// run mean-shift on all points
	mPointMode.assign(mNumPoints, -1);
	mPointBandwidth.assign(mNumPoints, -1);
#pragma omp parallel for
	for(int pointID=0; pointID<mNumPoints; pointID++) {
		if( !iterateMeanShift(pointID) ) {
			cout << "Error: iterating mean shift" << endl;
		}
		
	}

	// extract all modes
	set<int> modeSet;
	for (int pointID = 0; pointID < mNumPoints; pointID++) {
		if (mPointMode[pointID] >= 0) {
			int modeID = mPointMode[pointID];
			if (mPointBandwidth[modeID] >= 0) modeSet.insert(modeID);
			else mPointMode[pointID] = -1;
		}
	}
	vector<int> modeList(modeSet.begin(), modeSet.end());

	// sort by weight
	vector<double> pointWeights(mpDataWeights->begin(), mpDataWeights->end());
	sort(modeList.begin(), modeList.end(),
		[&pointWeights](const int lhs, const int rhs) { return pointWeights[lhs] > pointWeights[rhs]; });

	// combine modes
	vector<int> validModes;
	for (int possibleID : modeList) {
		double possibleRange = mPointBandwidth[possibleID] * StyleSimilarityConfig::mCluster_ModeMergingThreshold;
		bool merged = false;
		for (int validID : validModes) {
			double validRange = mPointBandwidth[validID] * StyleSimilarityConfig::mCluster_ModeMergingThreshold;
			double d = distance((*mpDataPoints)[possibleID], (*mpDataPoints)[validID]);
			if (d < possibleRange || d < validRange) {
				merged = true;
				break;
			}
		}
		if (!merged) {
			validModes.push_back(possibleID);
		}
	}
	modes.swap(validModes);

	map<int, int> modeIDMap;
	for (int id = 0; id < (int)modes.size(); id++) modeIDMap[modes[id]] = id;
	for (int &id : mPointMode) {
		if (modeIDMap.find(id) == modeIDMap.end()) {
			id = -1;
		} else {
			id = modeIDMap[id];
		}
	}
	modeIndices.assign(mPointMode.begin(), mPointMode.end());

	return true;
}

void ClusterMeanShift::clearUp() {

	if(mpKdTreeData) delete[] mpKdTreeData->ptr();
	mpKdTreeData = 0;
	mpKdTree = 0;
}

bool ClusterMeanShift::iterateMeanShift(int pointID) {

	// multiple iterations until convergence or divergence

	// first pass: gradient ascent
	//cout << "Mean-shift: first pass" << endl;

	vector<double> center((*mpDataPoints)[pointID].begin(), (*mpDataPoints)[pointID].end());
	double bandwidth = -1;
	{
		int numIteration = 0;
		bool converged = false;
		while (!converged && numIteration < StyleSimilarityConfig::mCluster_MeanShiftMaxIterations) {
			numIteration++;
			vector<double> newCenter;
			double newBandwidth;
			if (!doSingleMeanShift(center, newCenter, newBandwidth)) return false;
			if (distance(center, newCenter) <= newBandwidth * StyleSimilarityConfig::mCluster_MeanShiftConvergenceThreshold) {
				converged = true;
			}
			copy(newCenter.begin(), newCenter.end(), center.begin());
			bandwidth = newBandwidth;
		}
		if (!converged) {
			mPointMode[pointID] = -1;
			mPointBandwidth[pointID] = -1;
			return true;
		}
	}

	// second pass: check local maxima
	//cout << "Mean-shift: second pass" << endl;

	vector<double> originalCenter = center;
	if (!perturbCenter(originalCenter, center, bandwidth)) return false;

	{
		int numIteration = 0;
		bool converged = false;
		while (!converged && numIteration < StyleSimilarityConfig::mCluster_MeanShiftMaxIterations) {
			numIteration++;
			vector<double> newCenter;
			double newBandwidth;
			if (!doSingleMeanShift(center, newCenter, newBandwidth)) return false;
			if (distance(center, newCenter) <= newBandwidth * StyleSimilarityConfig::mCluster_MeanShiftConvergenceThreshold) {
				converged = true;
			}
			copy(newCenter.begin(), newCenter.end(), center.begin());
		}
		if (!converged) {
			mPointMode[pointID] = -1;
			mPointBandwidth[pointID] = -1;
			return true;
		}
	}

	if (distance(originalCenter, center) <= bandwidth * StyleSimilarityConfig::mCluster_ModeMergingThreshold * 0.5) { // UNDONE: param
		center = originalCenter;
	} else {
		// local maxima - ignored
		mPointMode[pointID] = -1;
		mPointBandwidth[pointID] = -1;
		return true;
	}

	// third pass: snap to existing point
	//cout << "Mean-shift: third pass" << endl;

	{
		vector<int> nbIndices;
		vector<double> nbDistances;
		if (!findRangeNeighbors(center, bandwidth, nbIndices, nbDistances)) return false; // inspect points within bandwidth
		if (nbIndices.empty()) {
			if (!findKNeighbors(center, 1, nbIndices, nbDistances)) return false; // just snap to nearest point
		}
		int bestID = -1;
		double bestWeight = 0;
		for(int idx : nbIndices) {
			double weight = (*mpDataWeights)[idx];
			if(weight > bestWeight) {
				bestWeight = weight;
				bestID = idx;
			}
			if (weight > mSufficientWeight) break;
		}
		
		mPointMode[pointID] = bestID;
		mPointBandwidth[pointID] = bandwidth;
	}

	return true;
}

bool ClusterMeanShift::doSingleMeanShift(vector<double> &inPoint, vector<double> &outPoint, double &outBandwidth) {

	// one iteration for mean shift

	if ((int)inPoint.size() != mDimensions) {
		cout << "Error: Incorrect dimension of points" << endl;
		return false;
	}

	// search for nearest neighbors within bandwidth

	double bandWidth;
	int kNeighbors;
	vector<int> vNbIndices;
	vector<double> vNbDistances;

	if(mFixedBandwidth > 0) {

		// use provided fixed bandwidth

		bandWidth = mFixedBandwidth;

		if( !findRangeNeighbors(inPoint, bandWidth, vNbIndices, vNbDistances) ) return false;
		kNeighbors = (int)vNbIndices.size();

	} else {

		// adaptive bandwidth estimated by cumulated weight

		kNeighbors = min(100, mNumPoints); // at most 100 neighbors
		if( !findKNeighbors(inPoint, kNeighbors, vNbIndices, vNbDistances) ) return false;
		
		double sumWeights = 0;
		for (int k = 0; k < kNeighbors; k++) {
			sumWeights += (*mpDataWeights)[vNbIndices[k]];
			//if (vNbDistances[k] && sumWeights > mSufficientWeight) {
			if (sumWeights > mSufficientWeight) {
				kNeighbors = k+1;
				break;
			}
		}

		bandWidth = kNeighbors ? vNbDistances[kNeighbors-1] : 0;
	}
	outBandwidth = bandWidth;

	// calculate mean-shift vector

	vector<double> totalWeightedValues(inPoint.size(), 0.0);
	double totalWeights = 0.0;

	for(int nbID=0; nbID<kNeighbors; nbID++) {

		vector<double> &nbPosition = (*mpDataPoints)[vNbIndices[nbID]];
		double nbWeight = (*mpDataWeights)[vNbIndices[nbID]];
		double xNormSq = cml::sqr( distance(nbPosition, inPoint) / bandWidth );
		double g = kernelDerivative(xNormSq);
				
		for(int d=0; d<mDimensions; d++) {
			totalWeightedValues[d] += nbWeight * g * (nbPosition[d]);
		}
		totalWeights += nbWeight * g;
	}

	// output new center

	if (totalWeights) {
		outPoint.resize(inPoint.size());
		for (int d = 0; d < mDimensions; d++) {
			outPoint[d] = totalWeightedValues[d] / totalWeights;
		}
	} else {
		outPoint = inPoint;
	}

	return true;
}

bool ClusterMeanShift::perturbCenter(vector<double> &inPoint, vector<double> &outPoint, double bandwidth) {

	// move point away with distance of bandwidth in random direction

	if ((int)inPoint.size() != mDimensions) {
		cout << "Error: Incorrect dimension of points" << endl;
		return false;
	}

	// copy
	outPoint = inPoint;

	// perturb
	double offset = cml::sqrt_safe(bandwidth * bandwidth / mDimensions) * 2.0;
	for (int d = 0; d<mDimensions; d++) {
		outPoint[d] += cml::random_real(-offset, offset);
	}

	return true;
}

bool ClusterMeanShift::findKNeighbors(vector<double> &inPoint, int k, vector<int> &outIndices, vector<double> &outDistances) {

	if( (int)inPoint.size() != mDimensions ) {
		cout << "Error: incorrect dimension of point" << endl;
		return false;
	}

	if( k > mNumPoints ) {
		cout << "Error: incorrect neighbor number" << endl;
		return false;
	}

	if( !mpKdTree ) {
		cout << "Error: not yet constructed a kD tree" << endl;
		return false;
	}

	flann::Matrix<double> queryData(new double[mDimensions], 1, mDimensions);
	flann::Matrix<int> queryIdx(new int[k], 1, k);
    flann::Matrix<double> queryDist(new double[k], 1, k);

	for(int i=0; i<mDimensions; i++) queryData[0][i] = inPoint[i];

	flann::SearchParams params;
	params.checks = -1;
	params.cores = 0;
	mpKdTree->knnSearch(queryData, queryIdx, queryDist, k, params);

	outIndices.resize(k);
	outDistances.resize(k);
	for(int i=0; i<k; i++) {
		outIndices[i] = queryIdx[0][i];
		outDistances[i] = cml::sqrt_safe( queryDist[0][i] ); // when using L2 distance the return of flann is distance square
	}

	delete[] queryData.ptr();
	delete[] queryIdx.ptr();
	delete[] queryDist.ptr();

	return true;
}

bool ClusterMeanShift::findRangeNeighbors(vector<double> &inPoint, double range, vector<int> &outIndices, vector<double> &outDistances) {

	if ((int)inPoint.size() != mDimensions) {
		cout << "Error: incorrect dimension of point" << endl;
		return false;
	}

	if( !mpKdTree ) {
		cout << "Error: not yet constructed a kD tree" << endl;
		return false;
	}

	flann::Matrix<double> queryData(new double[mDimensions], 1, mDimensions);
	vector<vector<size_t>> queryIdx(1);
	vector<vector<double>> queryDist(1);

	for(int i=0; i<mDimensions; i++) queryData[0][i] = inPoint[i];

	flann::SearchParams params;
	params.checks = -1;
	params.cores = 0;
	mpKdTree->radiusSearch(queryData, queryIdx, queryDist, (float)range, params);

	int numResults = (int)queryIdx[0].size();
	outIndices.resize(numResults);
	outDistances.resize(numResults);
	for(int i=0; i<numResults; i++) {
		outIndices[i] = (int)queryIdx[0][i];
		outDistances[i] = cml::sqrt_safe( queryDist[0][i] ); // when using L2 distance the return of flann is distance square
	}

	delete[] queryData.ptr();

	return true;
}

double ClusterMeanShift::kernelProfile(double xNormSq) {

	// Epanechnikov kernel
	return 1.0 - xNormSq;
}

double ClusterMeanShift::kernelDerivative(double xNormSq) {

	// Epanechnikov kernel
	return 1.0;
}

double ClusterMeanShift::distance(vector<double> &p1, vector<double> &p2) {

	if((int)p1.size() != mDimensions || (int)p2.size() != mDimensions) {
		cout << "Error: dimension incorrect when calculating distance" << endl;
		return 0.0;
	}

	// L2 distance
	double distSq = 0;
	for(int i=0; i<mDimensions; i++) {
		double deltaD = (p1[i]-p2[i]);
		distSq += deltaD * deltaD;
	}
	double dist = cml::sqrt_safe( distSq );

	return dist;
}