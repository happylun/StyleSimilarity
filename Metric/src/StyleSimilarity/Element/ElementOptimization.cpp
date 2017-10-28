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

#include "ElementOptimization.h"

#include <fstream>
#include <set>
#include <omp.h>

#include "Library/MAXFLOWHelper.h"
#include "Library/SidKDTreeHelper.h"

#include "Data/StyleSimilarityConfig.h"
#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"
#include "Segment/SegmentUtil.h"
#include "Element/ElementUtil.h"

#include "Match/MatchSimpleICP.h"

#include "Element/ElementMetric.h"

#define OUTPUT_PROGRESS

using namespace StyleSimilarity;

StyleSimilarityData* ElementOptimization::mpData = 0;

ElementOptimization::ElementOptimization(StyleSimilarityData *data) {

	mpData = data;
}

ElementOptimization::~ElementOptimization() {
}

bool ElementOptimization::process() {

#ifdef OUTPUT_PROGRESS
	cout << "Finding element..." << endl;
#endif

	mNumSourcePatches = (int)mpData->mSourcePatchesIndices.size();
	mNumTargetPatches = (int)mpData->mTargetPatchesIndices.size();
	mNumModes = (int)mpData->mTransformationModes.size();

	if (!findNearestPoints()) return false;
	if (!calculateEnergyTerms()) return false;
	if (!optimizeLabelAssignment()) return false;
	if (!extractElementParts()) return false;

#ifdef OUTPUT_PROGRESS
	cout << "Element done." << endl;
#endif

	return true;
}

bool ElementOptimization::findNearestPoints() {

#ifdef OUTPUT_PROGRESS
	cout << "Finding nearest points..." << endl;
#endif

	// compute patch map

	mSourcePatchMap.assign(mpData->mSourceSamples.amount, -1);
#pragma omp parallel for
	for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
		for (int sampleID : mpData->mSourcePatchesIndices[patchID]) {
			mSourcePatchMap[sampleID] = patchID;
		}
	}

	mTargetPatchMap.assign(mpData->mTargetSamples.amount, -1);
#pragma omp parallel for
	for (int patchID = 0; patchID < mNumTargetPatches; patchID++) {
		for (int sampleID : mpData->mTargetPatchesIndices[patchID]) {
			mTargetPatchMap[sampleID] = patchID;
		}
	}

	// find nearest points

	mSourceNearestPoint.resize(mpData->mSourceSamples.amount, mNumModes);
	mTargetNearestPoint.resize(mpData->mTargetSamples.amount, mNumModes);

	Eigen::Matrix3Xd matSP, matSN;
	Eigen::Matrix3Xd matTP, matTN;
	if (!SampleUtil::buildMatrices(mpData->mSourceSamples, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(mpData->mTargetSamples, matTP, matTN)) return false;

//#pragma omp parallel for
	for (int modeID = 0; modeID < mNumModes; modeID++) {

		Eigen::Affine3d transformation = mpData->mTransformationModes[modeID];
		auto rotation = transformation.rotation();

		Eigen::Matrix3Xd matXSP = transformation * matSP;
		SKDTree xspTree;
		SKDTreeData xspTreeData;
		if (!SampleUtil::buildKdTree(matXSP, xspTree, xspTreeData)) error("build KD tree");

		Eigen::VectorXi vecSourceIdx, vecTargetIdx;
		if (!SampleUtil::findNearestNeighbors(mpData->mTargetSamplesKdTree, matXSP, vecSourceIdx)) error("find NN");
		if (!SampleUtil::findNearestNeighbors(xspTree, matTP, vecTargetIdx)) error("find NN");

		// remove edge points & opposite points

		vector<int> tgtHits(mpData->mTargetSamples.amount, 0);
		for (int k = 0; k < (int)vecSourceIdx.size(); k++) {
			tgtHits[vecSourceIdx[k]]++;
		}
		for (int k = 0; k < (int)vecSourceIdx.size(); k++) {
			if (tgtHits[vecSourceIdx[k]] > 10) vecSourceIdx[k] = -1; // UNDONE: param edge point threshold
			if (vecSourceIdx[k] >= 0) {
				Eigen::Vector3d ns = rotation * matSN.col(k);
				Eigen::Vector3d nt = matTN.col(vecSourceIdx[k]);
				if (ns.dot(nt) <= 0) vecSourceIdx[k] = -1;
			}
		}

		vector<int> srcHits(mpData->mSourceSamples.amount, 0);
		for (int k = 0; k < (int)vecTargetIdx.size(); k++) {
			srcHits[vecTargetIdx[k]]++;
		}
		for (int k = 0; k < (int)vecTargetIdx.size(); k++) {
			if (srcHits[vecTargetIdx[k]] > 10) vecTargetIdx[k] = -1; // UNDONE: param edge point threshold
			if (vecTargetIdx[k] >= 0) {
				Eigen::Vector3d ns = rotation * matSN.col(vecTargetIdx[k]);
				Eigen::Vector3d nt = matTN.col(k);
				if (ns.dot(nt) <= 0) vecTargetIdx[k] = -1;
			}
		}

		mSourceNearestPoint.col(modeID) = vecSourceIdx;
		mTargetNearestPoint.col(modeID) = vecTargetIdx;
	}

	return true;
}

bool ElementOptimization::findMatchedPoints(vector<int> &inPatches, int inMode, vector<int> &outPoints) {

	vector<bool> sourcePatchFlag(mNumSourcePatches, false);
	vector<bool> targetPatchFlag(mNumTargetPatches, false);

	for (int sourcePatchID : inPatches) {
		sourcePatchFlag[sourcePatchID] = true;
		auto &sourcePatch = mpData->mSourcePatchesIndices[sourcePatchID];
#pragma omp parallel for
		for (int k = 0; k < (int)sourcePatch.size(); k++) {
			int targetPointID = mSourceNearestPoint(sourcePatch[k], inMode);
			if (targetPointID < 0) continue;
			int targetPatchID = mTargetPatchMap[targetPointID];
			if (targetPatchID >= 0) targetPatchFlag[targetPatchID] = true;
		}
	}

	outPoints.clear();
	for (int targetPatchID = 0; targetPatchID < mNumTargetPatches; targetPatchID++) {
		if (!targetPatchFlag[targetPatchID]) continue;
		auto &targetPatch = mpData->mTargetPatchesIndices[targetPatchID];

		int numPoints = (int)targetPatch.size();
		vector<bool> targetPointFlag(numPoints, false);
#pragma omp parallel for
		for (int k = 0; k < numPoints; k++) {
			int sourcePointID = mTargetNearestPoint(targetPatch[k], inMode);
			if (sourcePointID < 0) continue;
			int sourcePatchID = mSourcePatchMap[sourcePointID];
			if (sourcePatchID>=0 && sourcePatchFlag[sourcePatchID]) {
				targetPointFlag[k] = true;
			}
		}

		for (int k = 0; k < numPoints; k++) {
			if (targetPointFlag[k]) {
				outPoints.push_back(targetPatch[k]);
			}
		}
	}

	return true;
}

bool ElementOptimization::findMatchedPatches(vector<int> &inPatches, int inMode, vector<int> &outPatches) {

	vector<bool> sourcePatchFlag(mNumSourcePatches, false);
	vector<bool> targetPatchFlag(mNumTargetPatches, false);

	for (int sourcePatchID : inPatches) {
		sourcePatchFlag[sourcePatchID] = true;
		auto &sourcePatch = mpData->mSourcePatchesIndices[sourcePatchID];
#pragma omp parallel for
		for (int k = 0; k < (int)sourcePatch.size(); k++) {
			int targetPointID = mSourceNearestPoint(sourcePatch[k], inMode);
			if (targetPointID < 0) continue;
			int targetPatchID = mTargetPatchMap[targetPointID];
			if(targetPatchID>=0) targetPatchFlag[targetPatchID] = true;
		}
	}

#ifdef _OPENMP
	int numThreads = omp_get_max_threads();
#else
	int numThreads = 1;
#endif

	outPatches.clear();
	for (int targetPatchID = 0; targetPatchID < mNumTargetPatches; targetPatchID++) {
		if (!targetPatchFlag[targetPatchID]) continue;
		auto &targetPatch = mpData->mTargetPatchesIndices[targetPatchID];

		int totalCount = (int)targetPatch.size();
		vector<int> ompCounts(numThreads, 0);
#pragma omp parallel for num_threads(numThreads)
		for (int k = 0; k < totalCount; k++) {
			int sourcePointID = mTargetNearestPoint(targetPatch[k], inMode);
			if (sourcePointID < 0) continue;
			int sourcePatchID = mSourcePatchMap[sourcePointID];
			if (sourcePatchID>=0 && sourcePatchFlag[sourcePatchID]) {
#ifdef _OPENMP
				int threadID = omp_get_thread_num();
#else
				int threadID = 0;
#endif
				ompCounts[threadID]++;
			}
		}

		
		int matchCount = 0;
		for (int threadID = 0; threadID < numThreads; threadID++) {
			matchCount += ompCounts[threadID];
		}

		if (matchCount >= totalCount * StyleSimilarityConfig::mOptimization_MatchedPatchCoverage) {
			outPatches.push_back(targetPatchID);
		}
	}

	return true;
}

bool ElementOptimization::calculateEnergyTerms() {

#ifdef OUTPUT_PROGRESS
	cout << "Calculating energy terms" << endl;
#endif

	// initialize terms

	if (true) {
		mAssignedUnaryTerm.resize(mNumSourcePatches, mNumModes);
		mUnassignedUnaryTerm.resize(mNumSourcePatches, mNumModes);
		mAssignedUnaryTerm.setZero();
		mUnassignedUnaryTerm.setZero();

		set<vec2i> pairSet;
		for (int p1 = 0; p1 < mNumSourcePatches; p1++) {
			for (int p2 : mpData->mSourcePatchesGraph[p1]) {
				pairSet.insert(p1 < p2 ? vec2i(p1, p2) : vec2i(p2, p1));
			}
		}
		vector<vec2i> pairList(pairSet.begin(), pairSet.end());
		mNumAdjacentSourcePatches = (int)pairList.size();

		mPairwiseIndex.resize(mNumAdjacentSourcePatches, 2);
		for (int r = 0; r < mNumAdjacentSourcePatches; r++) {
			mPairwiseIndex(r, 0) = pairList[r][0];
			mPairwiseIndex(r, 1) = pairList[r][1];
		}
		mPairwiseTerm.resize(mNumAdjacentSourcePatches, mNumModes);
	}

	// compute patch centroid (used for estimate transformation in pairwise term)

	vector<vec3> sourcePatchCentroid(mNumSourcePatches);

#pragma omp parallel for
	for (int sourcePatchID = 0; sourcePatchID < mNumSourcePatches; sourcePatchID++) {
		TSampleSet &patch = mpData->mSourcePatches[sourcePatchID];
		vec3 bbMin, bbMax;
		if (!SampleUtil::computeAABB(patch, bbMin, bbMax)) error("compute patch AABB");
		sourcePatchCentroid[sourcePatchID] = (bbMin + bbMax) / 2;
	}

	// compute terms for each patch under each transformation

#pragma omp parallel for
	for (int modeID = 0; modeID < mNumModes; modeID++) {

		Eigen::Affine3d &transformation = mpData->mTransformationModes[modeID];

		double sigma = mpData->mDistanceSigma;
		if (modeID == 0) {
			sigma *= StyleSimilarityConfig::mOptimization_MegaSigmaFactor;
		}

		// compute unary terms

		//PlyExporter pes, pet;

		vector<bool> noMatchesFlag(mNumSourcePatches, false);

#pragma omp parallel for
		for (int sourcePatchID = 0; sourcePatchID < mNumSourcePatches; sourcePatchID++) {

			vector<int> sourcePatches(1, sourcePatchID);
			vector<int> sourcePoints = mpData->mSourcePatchesIndices[sourcePatchID];

			/*
			vector<int> targetPatches;
			if (!findMatchedPatches(sourcePatches, modeID, targetPatches)) error("find matched patches");

			if (targetPatches.empty()) {
				noMatchesFlag[sourcePatchID] = true;
				mAssignedUnaryTerm(sourcePatchID, modeID) = 0;
				mUnassignedUnaryTerm(sourcePatchID, modeID) = 0;
				continue;
			}
			
			vector<int> targetPoints(0);
			for (int targetPatchID : targetPatches) {
				auto &patch = mpData->mTargetPatchesIndices[targetPatchID];
				targetPoints.insert(targetPoints.end(), patch.begin(), patch.end());
			}
			*/
			
			vector<int> targetPoints;
			if (!findMatchedPoints(sourcePatches, modeID, targetPoints)) error("find matched points");
			if (targetPoints.size() < 20) {
				noMatchesFlag[sourcePatchID] = true;
				mAssignedUnaryTerm(sourcePatchID, modeID) = 0;
				mUnassignedUnaryTerm(sourcePatchID, modeID) = 0;
				continue;
			}
			

			Eigen::VectorXd distanceVec;
			if (!ElementMetric::computeSimplePatchDistance(
				mpData->mSourceSamples, mpData->mTargetSamples,
				sourcePoints, targetPoints,
				mpData->mpSourceFeatures, mpData->mpTargetFeatures,
				transformation, distanceVec)) error("compute patch distance");
			double distance = ElementMetric::evalSimplePatchDistance(distanceVec);
			
			double term1 = distance / sigma;
			double term0 = max(0.0, -log(1 - exp(-term1) + 1e-5));

			mAssignedUnaryTerm(sourcePatchID, modeID) = term1;
			mUnassignedUnaryTerm(sourcePatchID, modeID) = term0;
			
			/*
			if (modeID == 8) {
				if (sourcePatchID == 56) {
					cout << "Source " << sourcePatchID << " : " << distanceVec.topRows(2).transpose() << endl;
					cout << "Dist: " << distance << ", " << sigma << endl;
					cout << "Terms: " << term1 << ", " << term0 << endl;
				}
			}			
			
			if (modeID == 8) {
				TPointSet ps, pt;
				SegmentUtil::extractPointSet(mpData->mSourceSamples, sourcePoints, ps);
				SegmentUtil::extractPointSet(mpData->mTargetSamples, targetPoints, pt);
				matrix4f mat = matrix4ed(transformation.matrix().data());
				vec3i color = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
				pes.addPoint(&ps.positions, &ps.normals, mat, color);
				pet.addPoint(&pt.positions, &pt.normals, cml::identity_4x4(), color);
			}
			*/
		}

		if (true) {
			// compute unary term for unmatched patches
			double maxUnaryTerm = mAssignedUnaryTerm.col(modeID).maxCoeff();
			double unmatchedTerm0, unmatchedTerm1;
			if (maxUnaryTerm) {
				unmatchedTerm1 = maxUnaryTerm * StyleSimilarityConfig::mOptimization_UnmatchedUnaryFactor;
				unmatchedTerm0 = max(0.0, -log(1 - exp(-unmatchedTerm1) + 1e-5));
			} else {
				unmatchedTerm1 = 1e7;
				unmatchedTerm0 = 0;
			}
			
			for (int sourcePatchID = 0; sourcePatchID < mNumSourcePatches; sourcePatchID++) {
				if (noMatchesFlag[sourcePatchID]) {
					mAssignedUnaryTerm(sourcePatchID, modeID) = unmatchedTerm1;
					mUnassignedUnaryTerm(sourcePatchID, modeID) = unmatchedTerm0;
				}
			}
		}

		/*
		if (modeID == 8) {
			pes.output("Debug/3.transform/debug/building-11--199/ps.ply");
			pet.output("Debug/3.transform/debug/building-11--199/pt.ply");
			system("pause");
		}
		*/

		// compute pairwise terms
#pragma omp parallel for
		for (int pairID = 0; pairID < mNumAdjacentSourcePatches; pairID++) {

			int patchP = mPairwiseIndex(pairID, 0);
			int patchQ = mPairwiseIndex(pairID, 1);

			vector<int> pointsP = mpData->mSourcePatchesIndices[patchP];
			vector<int> pointsQ = mpData->mSourcePatchesIndices[patchQ];

			Eigen::Affine3d pairwiseTransformation; // only translation, no rotation or scaling
			pairwiseTransformation.setIdentity();
			vec3 offset = sourcePatchCentroid[patchQ] - sourcePatchCentroid[patchP];
			pairwiseTransformation.translate(Eigen::Vector3d(offset[0], offset[1], offset[2]));
			
			Eigen::VectorXd distanceVec;
			if (!ElementMetric::computeSimplePatchDistance(
				mpData->mSourceSamples, mpData->mSourceSamples,
				pointsP, pointsQ,
				mpData->mpSourceFeatures, mpData->mpSourceFeatures,
				pairwiseTransformation, distanceVec)) error("compute patch distance");
			double distance = ElementMetric::evalSimplePatchDistance(distanceVec);

			double term = max(0.0, -log(1 - exp(-distance/mpData->mDistanceSigma) + 1e-5));

			int numNeighborsPatch0 = (int)mpData->mSourcePatchesGraph[patchP].size();
			int numNeighborsPatch1 = (int)mpData->mSourcePatchesGraph[patchQ].size();
			int numNeighbors = max(1, (numNeighborsPatch0 + numNeighborsPatch1)/2);
			mPairwiseTerm(pairID, modeID) = term / numNeighbors;
			/*
			if (modeID == 1 && patchP == 0) {
				cout << "Pairwise term " << patchP << "-" << patchQ << ": " << term << endl;
				cout << "Unary term " << patchP << ": " << mAssignedUnaryTerm(patchP, modeID) << " " << mUnassignedUnaryTerm(patchP, modeID) << endl;
				cout << "Unary term " << patchQ << ": " << mAssignedUnaryTerm(patchQ, modeID) << " " << mUnassignedUnaryTerm(patchQ, modeID) << endl;
				cout << "distance: " << distance << endl;
				cout << "distances: " << distanceVec.topRows(2).transpose() << endl;
			}
			*/
		}
	}
	//system("pause");

	return true;
}

bool ElementOptimization::optimizeLabelAssignment() {

	mOptimizedLabels.clear();
	mOptimizedUnaryTerms.clear();
	mOptimizedPairwiseTerms.clear();
	if(mNumModes <= 0) {
		// no transformation modes found
		return true;
	}
	mOptimizedLabels.resize(mNumModes);
	mOptimizedUnaryTerms.resize(mNumModes);
	mOptimizedPairwiseTerms.resize(mNumModes);

#ifdef OUTPUT_PROGRESS
	cout << "Optimizing label assignment" << endl;
#endif

#pragma omp parallel for
	for (int modeID = 0; modeID < mNumModes; modeID++) {

		mOptimizedLabels[modeID].clear();
		mOptimizedLabels[modeID].resize(mNumSourcePatches, 0);

		/*
		if ((int)StyleSimilarityConfig::mData_CustomNumber3 == 44) {
			// label with ICP

			cout << "*";

			Eigen::Matrix3Xd matSP, matSN, matTP, matTN;
			SampleUtil::buildMatrices(mpData->mSourceSamples, matSP, matSN);
			SampleUtil::buildMatrices(mpData->mTargetSamples, matTP, matTN);
			Eigen::Affine3d mode = mpData->mTransformationModes[modeID];
			matSP = mode * matSP;
			matSN = mode.rotation() * matSN;
			vector<int> srcPoints, tgtPoints;
			MatchSimpleICP::prealign(matSP, matSN, matTP, matTN, &srcPoints, &tgtPoints);
			set<int> srcSet(srcPoints.begin(), srcPoints.end());
			set<int> tgtSet(tgtPoints.begin(), tgtPoints.end());
			for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
				int count = 0;
				int countAll = 0;
				for (int id : mpData->mSourcePatchesIndices[patchID]) {
					if (srcSet.find(id) != srcSet.end()) count++;
					countAll++;
				}
				if (count * 2 > countAll) {
					mOptimizedLabels[modeID][patchID] = true;
				}
				else {
					mOptimizedLabels[modeID][patchID] = false;
				}
			}

			continue;
		}
		
		if ((int)StyleSimilarityConfig::mData_CustomNumber3 == 99) {
			// use unary term only

			cout << "@";

			for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
				if (mAssignedUnaryTerm(patchID, modeID) <= mUnassignedUnaryTerm(patchID, modeID)) {
					mOptimizedLabels[modeID][patchID] = true;
				}
				else {
					mOptimizedLabels[modeID][patchID] = false;
				}
			}
			continue;
		}
		*/

		// build max-flow graph

		typedef Graph<double, double, double> TGraph;
		TGraph *graph = new TGraph(mNumSourcePatches, mNumAdjacentSourcePatches);
		graph->add_node(mNumSourcePatches);
		for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
			double tweightA = mAssignedUnaryTerm(patchID, modeID);
			double tweightU = mUnassignedUnaryTerm(patchID, modeID);
			graph->add_tweights(patchID, tweightA, tweightU);
		}
		for (int pairID = 0; pairID < mNumAdjacentSourcePatches; pairID++) {
			int p1 = mPairwiseIndex(pairID, 0);
			int p2 = mPairwiseIndex(pairID, 1);
			double eweight = mPairwiseTerm(pairID, modeID);
			graph->add_edge(p1, p2, eweight, eweight);
		}

		// run max-flow

		//cout << "Optimizing mode " << modeID << endl;
		//cout << "Before optimization: " << endl;
		//if (!verifyEnergyTerms(modeID)) error("compute energy");

		double flow = graph->maxflow();
		//cout << "Max flow: " << flow << endl;

		for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
			mOptimizedLabels[modeID][patchID] = (graph->what_segment(patchID) == TGraph::SINK);
		}

		delete graph;
		/*
		if (modeID == 8) {
			cout << "After optimization: " << endl;
			if (!verifyEnergyTerms(modeID)) error("compute energy");

			vector<bool> tmp = mOptimizedLabels[modeID];
			for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
				vec3 center = vec3(0.0f, 0.0f, 0.0f);
				for (vec3 v : mpData->mSourcePatches[patchID].positions) center += v;
				center /= (float)(mpData->mSourcePatches[patchID].positions.size());
				if (center[0] < -0.25f && center[1] > 0.2f) {

					mOptimizedLabels[modeID][patchID] = true;
					double va = mAssignedUnaryTerm(patchID, modeID);
					double vu = mUnassignedUnaryTerm(patchID, modeID);
					cout << "Patch " << patchID << " : " << va << ", " << vu << endl;
				}
			}
			cout << "Manual labeling: " << endl;
			if (!verifyEnergyTerms(modeID)) error("compute energy");
			system("pause");
			//mOptimizedLabels[modeID] = tmp;
		}
		*/
	}

	return true;
}

bool ElementOptimization::extractElementParts() {

#ifdef OUTPUT_PROGRESS
	cout << "Extracting element parts" << endl;
#endif

	mElementTransformations.clear();
	mSourceElementParts.clear();
	mTargetElementParts.clear();

	for (int modeID = 0; modeID < mNumModes; modeID++) {

		// find connected components within source labeled patches

		vector<bool> flags(mNumSourcePatches, false);
		for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
			if (mOptimizedLabels[modeID][patchID]) {
				flags[patchID] = true; // mark labeled patches only
			}
		}
		for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
			if (!flags[patchID]) continue;
			flags[patchID] = false;
			vector<int> sourcePatches;
			vector<int> queue(1, patchID);
			int head = 0;
			while (head < (int)queue.size()) { // BFS
				int nowPatch = queue[head];
				sourcePatches.push_back(nowPatch);
				for (int neighborPatch : mpData->mSourcePatchesGraph[nowPatch]) {
					if (flags[neighborPatch]) queue.push_back(neighborPatch);
					flags[neighborPatch] = false;
				}
				head++;
			}

			// find matched target patches

			vector<int> targetPatches;
			if (!findMatchedPatches(sourcePatches, modeID, targetPatches)) return false;
			if (!targetPatches.empty()) {
				mElementTransformations.push_back(modeID);
				mSourceElementParts.push_back(sourcePatches);
				mTargetElementParts.push_back(targetPatches);
			}
		}
	}

	mNumElements = (int)mElementTransformations.size();

	/*
	// adjust transformations
	vector<Eigen::Affine3d> newModes(mNumElements);
#pragma omp parallel for
	for (int eleID = 0; eleID < mNumElements; eleID++) {
		int modeID = mElementTransformations[eleID];
		Eigen::Affine3d transformation = mpData->mTransformationModes[modeID];
		vector<int> srcIdx, tgtIdx;
		for (int patchID : mSourceElementParts[eleID]) {
			for (int pointID : mpData->mSourcePatchesIndices[patchID]) {
				srcIdx.push_back(pointID);
			}
		}
		for (int patchID : mTargetElementParts[eleID]) {
			for (int pointID : mpData->mTargetPatchesIndices[patchID]) {
				tgtIdx.push_back(pointID);
			}
		}
		TPointSet srcPatch, tgtPatch;
		if (!SegmentUtil::extractPointSet(mpData->mSourceSamples, srcIdx, srcPatch)) error("extract src point");
		if (!SegmentUtil::extractPointSet(mpData->mTargetSamples, tgtIdx, tgtPatch)) error("extract tgt point");
		Eigen::Matrix3Xd srcP, srcN, tgtP, tgtN;
		if (!SampleUtil::buildMatrices(srcPatch, srcP, srcN)) error("build src matrices");
		if (!SampleUtil::buildMatrices(tgtPatch, tgtP, tgtN)) error("build tgt matrices");
		if (!MatchSimpleICP::run(20, srcP, srcN, tgtP, tgtN, transformation)) error("ICP");

		newModes[eleID] = transformation;
		mElementTransformations[eleID] = eleID;

#ifdef OUTPUT_PROGRESS
		cout << "\rAdjusting transformations " << (eleID+1) << " / " << mNumElements << "                ";
#endif
	}
	cout << endl;
	mpData->mTransformationModes.swap(newModes);
	mNumModes = mNumElements;
	*/

#ifdef OUTPUT_PROGRESS
	cout << "Extracted " << mNumElements << " elements" << endl;
#endif
	
	return true;
}

bool ElementOptimization::verifyEnergyTerms(int modeID) {

	// mainly for debug... actually I can get it from MAXFLOW

	// compute unary component in energy term
	double unaryComponent = 0;
	for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
		bool label = mOptimizedLabels[modeID][patchID];
		double uTerm = label ? mAssignedUnaryTerm(patchID, modeID) : mUnassignedUnaryTerm(patchID, modeID);
		unaryComponent += uTerm;
	}

	// compute pairwise component in energy term
	double pairwiseComponent = 0;
	for (int pairID = 0; pairID < mNumAdjacentSourcePatches; pairID++) {
		bool label1 = mOptimizedLabels[modeID][mPairwiseIndex(pairID, 0)];
		bool label2 = mOptimizedLabels[modeID][mPairwiseIndex(pairID, 1)];
		double pTerm = label1 == label2 ? 0.0 : mPairwiseTerm(pairID, modeID);
		pairwiseComponent += pTerm;
	}

	double totalEnergy = unaryComponent + pairwiseComponent;
	cout << "Computed energy: " << unaryComponent << ", " << pairwiseComponent << ", " << totalEnergy << endl;

	mOptimizedUnaryTerms[modeID] = unaryComponent;
	mOptimizedPairwiseTerms[modeID] = pairwiseComponent;

	return true;
}

bool ElementOptimization::output(string elementFileName) {

#ifdef OUTPUT_PROGRESS
	cout << "Exporting element results" << endl;
#endif

	ofstream elementFile(elementFileName);

	elementFile << mNumElements << endl;

	for (int eleID = 0; eleID < mNumElements; eleID++) {

		elementFile << mElementTransformations[eleID] << endl;

		elementFile << mSourceElementParts[eleID].size();
		for (int id : mSourceElementParts[eleID]) {
			elementFile << " " << id;
		}
		elementFile << endl;

		elementFile << mTargetElementParts[eleID].size();
		for (int id : mTargetElementParts[eleID]) {
			elementFile << " " << id;
		}
		elementFile << endl;
	}

	elementFile.close();

	return true;
}

bool ElementOptimization::visualize(string pathName, string affix) {

#ifdef OUTPUT_PROGRESS
	cout << "Visualizing element results" << endl;
#endif

	// get bounding box
	vec3 bbScale;
	if (true) {
		vec3 bbMinS, bbMaxS, bbMinT, bbMaxT;
		if (!SampleUtil::computeAABB(mpData->mSourceSamples, bbMinS, bbMaxS)) return false;
		if (!SampleUtil::computeAABB(mpData->mTargetSamples, bbMinT, bbMaxT)) return false;
		vec3 bbScaleS = bbMaxS - bbMinS;
		vec3 bbScaleT = bbMaxT - bbMinT;
		bbScale = bbScaleS;
		bbScale.maximize(bbScaleT);
		bbScale *= 1.1f;
	}

	// visualize label
	if(true) {
		PlyExporter pe;
		for (int modeID = 0; modeID < mNumModes; modeID++) {
			vec3 vOffset = vec3(float(modeID % 4) * bbScale[0], 0.0f, -float(modeID / 4) * bbScale[2]);
			for (int patchID = 0; patchID < mNumSourcePatches; patchID++) {
				auto &patchP = mpData->mSourcePatches[patchID].positions;
				auto &patchN = mpData->mSourcePatches[patchID].normals;
				if (mOptimizedLabels[modeID][patchID]) {
					if (!pe.addPoint(&patchP, &patchN, vOffset, vec3i(255, 0, 0))) return false;
				} else {
					if (!pe.addPoint(&patchP, &patchN, vOffset, vec3i(127, 127, 127))) return false;
				}
			}
		}
		if (!pe.output(pathName + "/vis-sourceLabel" + affix + ".ply")) return false;
	}

	// visualize modes
	if(true) {
		PlyExporter pe;
		for (int modeID = 0; modeID < mNumModes; modeID++) {

			Eigen::Affine3d transformation = mpData->mTransformationModes[modeID];

			vec3 vOffset = vec3(float(modeID % 4) * bbScale[0], 0.0f, -float(modeID / 4) * bbScale[2]);
			transformation.pretranslate(Eigen::Vector3d(vOffset[0], vOffset[1], vOffset[2]));
			matrix4ed matFinalE(transformation.matrix().data());
			matrix4f matFinal = matFinalE;

			if (!pe.addPoint(&mpData->mSourceSamples.positions, &mpData->mSourceSamples.normals, matFinal, vec3i(127, 255, 255))) return false;
			if (!pe.addPoint(&mpData->mTargetSamples.positions, &mpData->mTargetSamples.normals, vOffset, vec3i(127, 127, 127))) return false;
		}
		if (!pe.output(pathName + "/vis-transformModes" + affix + ".ply")) return false;
	}

	// visualize element parts
	if (true) {
		// establish order (simply order by patch amount)
		vector<int> elementSize(mNumElements);
		for (int eleID = 0; eleID < mNumElements; eleID++) {
			elementSize[eleID] = (int)(mSourceElementParts[eleID].size() + mTargetElementParts[eleID].size());
		}
		vector<int> elementOrder(mNumElements);
		for (int j = 0; j < mNumElements; j++) elementOrder[j] = j;
		sort(elementOrder.begin(), elementOrder.end(),
			[&elementSize](int i1, int i2) { return elementSize[i1] > elementSize[i2]; });

		// export visualization
		PlyExporter pe;
		stringstream ss;
		for (int eleID = 0; eleID < mNumElements; eleID++) {

			//int partID = elementOrder[eleID];
			int partID = eleID;
			ss << partID << " ";

			// export visualization
			vec3 srcOffset = vec3(float(eleID % 4) * bbScale[0] * 2, 0.0f, -float(eleID / 4) * bbScale[2] * 2);
			vec3 tgtOffset = srcOffset + vec3(bbScale[0], 0.0f, 0.0f);

			for (int patchID : mSourceElementParts[partID]) {
				auto &patch = mpData->mSourcePatches[patchID];
				if (!pe.addPoint(&patch.positions, &patch.normals, srcOffset, vec3i(255, 0, 0))) return false;
			}
			for (int patchID : mTargetElementParts[partID]) {
				auto &patch = mpData->mTargetPatches[patchID];
				if (!pe.addPoint(&patch.positions, &patch.normals, tgtOffset, vec3i(0, 0, 255))) return false;
			}

			if (!pe.addPoint(&mpData->mSourceSamples.positions, &mpData->mSourceSamples.normals, srcOffset, vec3i(127, 127, 127))) return false;
			if (!pe.addPoint(&mpData->mTargetSamples.positions, &mpData->mTargetSamples.normals, tgtOffset, vec3i(127, 127, 127))) return false;
		}
		if (!pe.addComment(ss.str())) return false;
		if (!pe.output(pathName + "/vis-element" + affix + ".ply")) return false;
	}

	return true;
}

void ElementOptimization::error(string s) {
	cout << "Error: " << s << endl;
}
