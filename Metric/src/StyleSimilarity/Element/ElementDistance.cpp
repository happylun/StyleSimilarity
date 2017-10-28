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

#include "ElementDistance.h"

#include <fstream>
#include <algorithm>

#include "Element/ElementUtil.h"
#include "Element/ElementMetric.h"

#include "Match/MatchSimpleICP.h"

#include "Utility/FileUtil.h"

#include "Data/StyleSimilarityConfig.h"

#define OUTPUT_PROGRESS

using namespace StyleSimilarity;

ElementDistance::ElementDistance(StyleSimilarityData *data) {

	mpData = data;
}

ElementDistance::~ElementDistance() {
}

bool ElementDistance::process(string path, string affix) {

	string metricPDFileName = path + "data-metric-pd" + affix + ".txt";
	string metricSDFileName = path + "data-metric-sd" + affix + ".txt";
	string elementASFileName = path + "data-index-as" + affix + ".txt";
	string elementATFileName = path + "data-index-at" + affix + ".txt";
	string elementUSFileName = path + "data-index-us" + affix + ".txt";
	string elementUTFileName = path + "data-index-ut" + affix + ".txt";

	if (!FileUtil::existsfile(metricPDFileName)) {
		if (!computePatchDistance()) return false;
		if (!ElementUtil::saveMatrixBinary(metricPDFileName, mPatchDistance)) return false;
	}

	if (!FileUtil::existsfile(metricSDFileName)) {
		if (!computeShapeDistance()) return false;
		if (!ElementUtil::saveRowVectorBinary(metricSDFileName, mShapeDistance)) return false;
	}

	if (!ElementUtil::saveCellArraysBinary(elementASFileName, mElementSourcePoints)) return false;
	if (!ElementUtil::saveCellArraysBinary(elementATFileName, mElementTargetPoints)) return false;
	if (!ElementUtil::saveCellArrayBinary(elementUSFileName, mUnmatchSourcePoints)) return false;
	if (!ElementUtil::saveCellArrayBinary(elementUTFileName, mUnmatchTargetPoints)) return false;

	/*

	string metricGDFileName = path + "data-metric-gd" + affix + ".txt";
	string elementGSFileName = path + "data-patch-gs" + affix + ".txt";
	string elementGTFileName = path + "data-patch-gt" + affix + ".txt";

	string metricDFileName = path + "data-patch-d" + affix + ".txt";
	string elementASFileName = path + "data-patch-as" + affix + ".txt";
	string elementATFileName = path + "data-patch-at" + affix + ".txt";
	string elementUSFileName = path + "data-patch-us" + affix + ".txt";
	string elementUTFileName = path + "data-patch-ut" + affix + ".txt";


	if (!FileUtil::existsfile(metricPDFileName)) {
		if (!computePatchDistance()) return false;
		if (!ElementUtil::saveMatrixBinary(metricPDFileName, mPatchDistance)) return false;
	} else {
		if (!ElementUtil::loadMatrixBinary(metricPDFileName, mPatchDistance)) return false;
	}

	if (!FileUtil::existsfile(metricGDFileName)) {
	//if (true) {
		if (!computeShapeDistance()) return false;
		if (!ElementUtil::saveRowVectorBinary(metricGDFileName, mShapeDistance)) return false;
		if (!ElementUtil::saveCellArrayBinary(elementGSFileName, mSourceGlobalPatch)) return false;
		if (!ElementUtil::saveCellArrayBinary(elementGTFileName, mTargetGlobalPatch)) return false;
	} else {
		if (!ElementUtil::loadRowVectorBinary(metricGDFileName, mShapeDistance)) return false;
		if (!ElementUtil::loadCellArrayBinary(elementGSFileName, mSourceGlobalPatch)) return false;
		if (!ElementUtil::loadCellArrayBinary(elementGTFileName, mTargetGlobalPatch)) return false;
	}
	
	if (!FileUtil::existsfile(metricDFileName)) {
	//if (true) {

		Eigen::MatrixXd matD(mPatchDistance.rows() + 1, mShapeDistance.cols());
		if(mPatchDistance.rows()) matD.topRows(mPatchDistance.rows()) = mPatchDistance;
		matD.bottomRows(1) = mShapeDistance;

		mElementSourcePoints.push_back(mSourceGlobalPatch);
		mElementTargetPoints.push_back(mTargetGlobalPatch);

		
		//sort(mUnmatchSourcePoints.begin(), mUnmatchSourcePoints.end());
		//sort(mUnmatchTargetPoints.begin(), mUnmatchTargetPoints.end());
		//sort(mSourceGlobalPatch.begin(), mSourceGlobalPatch.end());
		//sort(mTargetGlobalPatch.begin(), mTargetGlobalPatch.end());
		//vector<int> outUnmatchSource(mUnmatchSourcePoints.size() + mSourceGlobalPatch.size());
		//vector<int> outUnmatchTarget(mUnmatchTargetPoints.size() + mTargetGlobalPatch.size());
		//auto itSource = set_difference(mUnmatchSourcePoints.begin(), mUnmatchSourcePoints.end(),
		//	mSourceGlobalPatch.begin(), mSourceGlobalPatch.end(), outUnmatchSource.begin());
		//outUnmatchSource.resize(itSource - outUnmatchSource.begin());
		//auto itTarget = set_difference(mUnmatchTargetPoints.begin(), mUnmatchTargetPoints.end(),
		//	mTargetGlobalPatch.begin(), mTargetGlobalPatch.end(), outUnmatchTarget.begin());
		//outUnmatchTarget.resize(itTarget - outUnmatchTarget.begin());
		

		if (!ElementUtil::saveMatrixBinary(metricDFileName, matD)) return false;
		if (!ElementUtil::saveCellArraysBinary(elementASFileName, mElementSourcePoints)) return false;
		if (!ElementUtil::saveCellArraysBinary(elementATFileName, mElementTargetPoints)) return false;
		//if (!ElementUtil::saveCellArrayBinary(elementUSFileName, outUnmatchSource)) return false;
		//if (!ElementUtil::saveCellArrayBinary(elementUTFileName, outUnmatchTarget)) return false;
		if (!ElementUtil::saveCellArrayBinary(elementUSFileName, mUnmatchSourcePoints)) return false;
		if (!ElementUtil::saveCellArrayBinary(elementUTFileName, mUnmatchTargetPoints)) return false;
	}

	*/

	return true;
}

bool ElementDistance::loadElement(string elementFileName) {

#ifdef OUTPUT_PROGRESS
	cout << "Loading element..." << endl;
#endif

	// parse element file

	ifstream eleFile(elementFileName);

	eleFile >> mNumElements;
	mElementSourcePoints.resize(mNumElements, vector<int>(0));
	mElementTargetPoints.resize(mNumElements, vector<int>(0));
	mElementTransformations.resize(mNumElements, -1);

	vector<bool> sourcePointFlags(mpData->mSourceSamples.amount, false);
	vector<bool> targetPointFlags(mpData->mTargetSamples.amount, false);

	for (int eleID = 0; eleID < mNumElements; eleID++) {

		eleFile >> mElementTransformations[eleID];
		bool isElement = (mElementTransformations[eleID] > 0); // skip mega element in prevalence term
		//bool isElement = true; // use all elements in prevalence term

		int srcNumPatches;
		eleFile >> srcNumPatches;
		for (int j = 0; j < srcNumPatches; j++) {
			int patchID;
			eleFile >> patchID;
			for (int sampleID : mpData->mSourcePatchesIndices[patchID]) {
				mElementSourcePoints[eleID].push_back(sampleID);
				if(isElement) sourcePointFlags[sampleID] = true;
			}
		}

		int tgtNumPatches;
		eleFile >> tgtNumPatches;
		for (int j = 0; j < tgtNumPatches; j++) {
			int patchID;
			eleFile >> patchID;
			for (int sampleID : mpData->mTargetPatchesIndices[patchID]) {
				mElementTargetPoints[eleID].push_back(sampleID);
				if(isElement) targetPointFlags[sampleID] = true;
			}
		}
	}
	eleFile.close();

	mUnmatchSourcePoints.clear();
	for (int pointID = 0; pointID < mpData->mSourceSamples.amount; pointID++) {
		if (!sourcePointFlags[pointID]) mUnmatchSourcePoints.push_back(pointID);
	}

	mUnmatchTargetPoints.clear();
	for (int pointID = 0; pointID < mpData->mTargetSamples.amount; pointID++) {
		if (!targetPointFlags[pointID]) mUnmatchTargetPoints.push_back(pointID);
	}

	return true;
}

bool ElementDistance::computePatchDistance() {

#ifdef OUTPUT_PROGRESS
	cout << "Computing patch distance..." << endl;
#endif

	if (mNumElements == 0) {
		// no element found
		mPatchDistance.resize(0, 0);
		return true;
	}

	vector<vector<double>> tmpDistance(mNumElements);
//#pragma omp parallel for
	for (int eleID = 0; eleID < mNumElements; eleID++) {
		Eigen::VectorXd distance;
		if (!ElementMetric::computeFullPatchDistance(
			mpData->mSourceSamples, mpData->mTargetSamples,
			mElementSourcePoints[eleID], mElementTargetPoints[eleID],
			*(mpData->mpSourceFeatures), *(mpData->mpTargetFeatures),
			mpData->mTransformationModes[mElementTransformations[eleID]],
			distance)) error("compute patch distance");
		int dim = (int)distance.size();
		tmpDistance[eleID].resize(dim);
		for (int d = 0; d < dim; d++) tmpDistance[eleID][d] = distance(d);
	}

	int dim = (int)tmpDistance[0].size();
	mPatchDistance.resize(mNumElements, dim);
	for (int r = 0; r < mNumElements; r++) {
		for (int c = 0; c < dim; c++) {
			mPatchDistance(r, c) = tmpDistance[r][c];
		}
	}

	return true;
}

bool ElementDistance::computeShapeDistance() {

#ifdef OUTPUT_PROGRESS
	cout << "Computing shape distance..." << endl;
#endif

	Eigen::Affine3d transformation = mpData->mTransformationModes[0];

	Eigen::VectorXd distance;
	if (!ElementMetric::computeFullShapeDistance(
		mpData->mSourceSamples, mpData->mTargetSamples,
		*(mpData->mpSourceFeatures), *(mpData->mpTargetFeatures),
		transformation, distance)) return false;

	mShapeDistance = distance;

	/*
	
	//Eigen::Matrix3Xd matSP, matSN, matTP, matTN;
	//if (!SampleUtil::buildMatrices(mpData->mSourceSamples, matSP, matSN)) return false;
	//if (!SampleUtil::buildMatrices(mpData->mTargetSamples, matTP, matTN)) return false;
	//matSP = transformation * matSP;
	//matSN = transformation.rotation() * matSN;
	//if (!MatchSimpleICP::prealign(matSP, matSN, matTP, matTN, &mSourceGlobalPatch, &mTargetGlobalPatch)) return false;
	
	int numSrc = mpData->mSourceSamples.amount;
	int numTgt = mpData->mTargetSamples.amount;
	mSourceGlobalPatch.resize(numSrc);
	mTargetGlobalPatch.resize(numTgt);
	for (int k = 0; k < numSrc; k++) mSourceGlobalPatch[k] = k;
	for (int k = 0; k < numTgt; k++) mTargetGlobalPatch[k] = k;

	Eigen::VectorXd distance;
	if (!ElementMetric::computeFullPatchDistance(
		mpData->mSourceSamples, mpData->mTargetSamples,
		mSourceGlobalPatch, mTargetGlobalPatch,
		*(mpData->mpSourceFeatures), *(mpData->mpTargetFeatures),
		transformation, distance)) return false;

	mShapeDistance = distance;

	*/

	return true;
}

void ElementDistance::error(string s) {
	cout << "Error: " << s << endl;
}