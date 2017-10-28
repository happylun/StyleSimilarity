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

#include "ElementVoting.h"

#include <iostream>
#include <fstream>

#include <Eigen/Eigen>

#include "Mesh/MeshUtil.h"

#include "Sample/SampleUtil.h"
#include "Segment/SegmentUtil.h"
#include "Element/ElementUtil.h"

#include "Match/MatchSimpleICP.h"
#include "Cluster/ClusterMeanShift.h"
#include "Element/ElementMetric.h"

#include "Data/StyleSimilarityConfig.h"

#include "Utility/PlyLoader.h"
#include "Utility/PlyExporter.h"

using namespace StyleSimilarity;

ElementVoting::ElementVoting(StyleSimilarityData *data) {

	mpData = data;
}

ElementVoting::~ElementVoting() {

}

bool ElementVoting::computeVotes() {

	cout << "Computing votes..." << endl;

	int numSourceSegments = (int)mpData->mSourceSegments.size();
	int numTargetSegments = (int)mpData->mTargetSegments.size();
	int numPairs = numSourceSegments * numTargetSegments;

	double sourceRadius = mpData->mSourceSamples.radius;
	double targetRadius = mpData->mTargetSamples.radius;

	// matrices

	vector<Eigen::Matrix3Xd> matSourcePosition(numSourceSegments);
	vector<Eigen::Matrix3Xd> matSourceNormal(numSourceSegments);
	vector<Eigen::Matrix3Xd> matSourceSubP(numSourceSegments);
	vector<Eigen::Matrix3Xd> matSourceSubN(numSourceSegments);

	vector<Eigen::Matrix3Xd> matTargetPosition(numTargetSegments);
	vector<Eigen::Matrix3Xd> matTargetNormal(numTargetSegments);
	vector<Eigen::Matrix3Xd> matTargetSubP(numTargetSegments);
	vector<Eigen::Matrix3Xd> matTargetSubN(numTargetSegments);

	// intrinsics (for preliminary filtering)

	vector<double> vecSourceVolume(numSourceSegments);
	vector<Eigen::Vector3d> vecSourceVariance(numSourceSegments);

	vector<double> vecTargetVolume(numTargetSegments);
	vector<Eigen::Vector3d> vecTargetVariance(numTargetSegments);

#pragma omp parallel for
	for (int sourceID = 0; sourceID < numSourceSegments; sourceID++) {

		if (mpData->mSourceSegments[sourceID].amount < 20) continue;

		if (!SampleUtil::buildMatrices(
			mpData->mSourceSegments[sourceID],
			matSourcePosition[sourceID],
			matSourceNormal[sourceID])) error("build matrix for source");

		Eigen::VectorXi subIdx;
		if (!SampleUtil::subSampleMatrices(
			matSourcePosition[sourceID],
			subIdx, 1000)) error("subsample source: index");
		if (!SampleUtil::sliceMatrices(
			matSourcePosition[sourceID], subIdx,
			matSourceSubP[sourceID])) error("subsample source: position");
		if (!SampleUtil::sliceMatrices(
			matSourceNormal[sourceID], subIdx,
			matSourceSubN[sourceID])) error("subsample source: normal");

		if (!computePatchIntrinsics(
			matSourceSubP[sourceID],
			sourceRadius,
			vecSourceVolume[sourceID],
			vecSourceVariance[sourceID])) error("source intrinsics");
	}

#pragma omp parallel for
	for (int targetID = 0; targetID < numTargetSegments; targetID++) {

		if (mpData->mTargetSegments[targetID].amount < 20) continue;

		if (!SampleUtil::buildMatrices(
			mpData->mTargetSegments[targetID],
			matTargetPosition[targetID],
			matTargetNormal[targetID])) error("build matrix for target");

		Eigen::VectorXi subIdx;
		if (!SampleUtil::subSampleMatrices(
			matTargetPosition[targetID],
			subIdx, 1000)) error("subsample target: index");
		if (!SampleUtil::sliceMatrices(
			matTargetPosition[targetID], subIdx,
			matTargetSubP[targetID])) error("subsample target: position");
		if (!SampleUtil::sliceMatrices(
			matTargetNormal[targetID], subIdx,
			matTargetSubN[targetID])) error("subsample target: normal");

		if (!computePatchIntrinsics(
			matTargetSubP[targetID],
			targetRadius,
			vecTargetVolume[targetID],
			vecTargetVariance[targetID])) error("target intrinsics");
	}

	// run ICP for every pair of source-target segments

	const int numMaxVotesPerPair = 5;
	int numVotes = numPairs*numMaxVotesPerPair;
	mTransformationVotes.resize(numVotes);

	int pairCount = 0;
#pragma omp parallel for shared(pairCount)
	for (int voteID = 0; voteID < numPairs; voteID++) {

		int sourceID = voteID / numTargetSegments;
		int targetID = voteID % numTargetSegments;
		/*
		// for debug
		if (sourceID < (int)(StyleSimilarityConfig::mData_CustomNumber1)
			|| targetID < (int)(StyleSimilarityConfig::mData_CustomNumber2)) continue;
			*/
		if (mpData->mSourceSegments[sourceID].amount < 20) continue;
		if (mpData->mTargetSegments[targetID].amount < 20) continue;
		/*
		if (!SampleUtil::saveSample(
			"D:/SAM/Data/Debug/3.transform/debug/lamp-L1-1--L1-3/source.ply",
			mpData->mSourceSegments[sourceID])) error("save source patch");
		if (!SampleUtil::saveSample(
			"D:/SAM/Data/Debug/3.transform/debug/lamp-L1-1--L1-3/target.ply",
			mpData->mTargetSegments[targetID])) error("save target patch");
			*/
		if (!checkPatchPairs( vecSourceVolume[sourceID], vecTargetVolume[targetID],
			vecSourceVariance[sourceID], vecTargetVariance[targetID])) continue;

		if (pairCount % 100 == 0) cout << "\rMatching pair " << pairCount << " / " << numPairs << "          ";
#pragma omp atomic
		pairCount++;

		// run ICP
		vector<double> allErrors;
		vector<Eigen::Affine3d> allTransformations;
		double bestError = DBL_MAX;
		for (int i = 0; i <= 4; i++) {
			// initial transformation		
			// 0~3: rotate 90 degrees 4 times along up axis
			// 4: align OBB direction

			Eigen::Affine3d transformation;
			transformation.setIdentity();

			if (i < 4) {
				transformation.rotate(Eigen::AngleAxisd(0.5*M_PI*i, Eigen::Vector3d::UnitY()));
			} else {
				Eigen::Matrix3d sourceCS, targetCS;
				if (!MatchSimpleICP::preorient(matSourcePosition[sourceID], sourceCS)) error("preorient source");
				if (!MatchSimpleICP::preorient(matTargetPosition[targetID], targetCS)) error("preorient target");
				transformation = targetCS.transpose() * sourceCS;
			}
			double errorICP;
			
			if (!MatchSimpleICP::run(20, // UNDONE: param ICP iteration for voting
				matSourceSubP[sourceID],
				matSourceSubN[sourceID],
				matTargetSubP[targetID],
				matTargetSubN[targetID],
				transformation)) error("ICP for voting");
			/*
			if (!MatchSimpleICP::visualize(
				"D:/SAM/Data/Debug/3.transform/debug/lamp-L1-1--L1-3/ICP.ply",
				matSourcePosition[sourceID], matTargetPosition[targetID], transformation)) error("visualize ICP");
			system("pause");
			*/
			Eigen::Matrix3Xd matSourceTransformed = transformation * matSourceSubP[sourceID];
			if (!MatchSimpleICP::error(
				matSourceTransformed,
				matTargetSubP[targetID],
				errorICP)) error("compute ICP error");

			allErrors.push_back(errorICP);
			allTransformations.push_back(transformation);
			if (errorICP < bestError) {
				bestError = errorICP;
			}
		}

		for (int k = 0; k < (int)allTransformations.size(); k++) {

			// generate votes
			auto &vote = mTransformationVotes[voteID*numMaxVotesPerPair + k];
			vote.patchSourceID = sourceID;
			vote.patchTargetID = targetID;
			vote.transformation = allTransformations[k];
			vote.weight = 1.0; // will compute later. just mark it as valid
		}
	}
	cout << endl;
	numPairs = pairCount;

	// prune invalid votes
#pragma omp parallel for
	for (int voteID = 0; voteID < numVotes; voteID++) {
		auto &vote = mTransformationVotes[voteID];
		if (vote.weight == 0) continue;
		if (!vote.transformation.matrix().allFinite()) {
			vote.weight = 0;
			continue;
		}
		vector<double> trs;
		ElementUtil::convertAffineToTRS(vote.transformation, trs);
		for (int j = 0; j < 3; j++) {
			double s = fabs(trs[7 + j]);
			if (s > 5.0 || s < 0.2) { // UNDONE: param max/min scaling factor
				vote.weight = 0;
			}
		}
	}
	auto &vec = mTransformationVotes; // shorter name...
	vec.erase(remove(vec.begin(), vec.end(), TTransformVote()), vec.end());
	numVotes = (int)mTransformationVotes.size();
	cout << "Number of pairs: " << numPairs << endl;
	cout << "Number of votes: " << numVotes << endl;

	return true;
}

bool ElementVoting::computeVoteDistances() {

	cout << "Computing vote distances..." << endl;

	int numVotes = (int)mTransformationVotes.size();
	if (numVotes == 0) { // no votes...
		mTransformationVoteDistance.resize(0, 0);
		return true;
	}
	vector<vector<double>> tmpDistance(numVotes);

#pragma omp parallel for
	for (int voteID = 0; voteID < numVotes; voteID++) {
		TTransformVote &vote = mTransformationVotes[voteID];

		Eigen::VectorXd distance;
		if (!ElementMetric::computeSimplePatchDistance(
			mpData->mSourceSamples, mpData->mTargetSamples,
			mpData->mSourceSegmentsIndices[vote.patchSourceID],
			mpData->mTargetSegmentsIndices[vote.patchTargetID],
			mpData->mpSourceFeatures, mpData->mpTargetFeatures,
			vote.transformation, distance)) error("compute patch distance");
		
		int dim = (int)distance.size();
		tmpDistance[voteID].resize(dim);
		for (int d = 0; d < dim; d++) tmpDistance[voteID][d] = distance(d);
	}

	int dim = (int)tmpDistance[0].size();
	mTransformationVoteDistance.resize(numVotes, dim);
	for (int r = 0; r < numVotes; r++) {
		for (int c = 0; c < dim; c++) {
			mTransformationVoteDistance(r, c) = tmpDistance[r][c];
		}
	}

	return true;
}

bool ElementVoting::computeGlobalAlignment() {

	cout << "Finding global alignment" << endl;

	// ICP on entire shape

	Eigen::Matrix3Xd matSP, matSN, matTP, matTN;
	if (!SampleUtil::buildMatrices(mpData->mSourceSamples, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(mpData->mTargetSamples, matTP, matTN)) return false;

	mGlobalTransformation.setIdentity();
	if (!MatchSimpleICP::run(20,
		matSP, matSN, matTP, matTN,
		mGlobalTransformation)) return false;

	// simple patch distance

	vector<int> sourceIdx(mpData->mSourceSamples.amount);
	vector<int> targetIdx(mpData->mTargetSamples.amount);
	for (int k = 0; k < (int)sourceIdx.size(); k++) sourceIdx[k] = k;
	for (int k = 0; k < (int)targetIdx.size(); k++) targetIdx[k] = k;

	if (!ElementMetric::computeSimplePatchDistance(
		mpData->mSourceSamples, mpData->mTargetSamples,
		sourceIdx, targetIdx,
		mpData->mpSourceFeatures, mpData->mpTargetFeatures,
		mGlobalTransformation, mGlobalDistance)) return false;

	return true;
}

bool ElementVoting::clusterVotes(double sigma) {

	if (mTransformationVotes.empty()) { // no transformation found
		mpData->mTransformationModes.clear();
		return true;
	}
	int numVotes = (int)mTransformationVotes.size();

	cout << "Clustering votes..." << endl;

	// evaluate vote distances
	Eigen::VectorXd voteDistances(numVotes);
	for (int voteID = 0; voteID < numVotes; voteID++) {
		Eigen::VectorXd distanceVec = mTransformationVoteDistance.row(voteID).transpose();
		voteDistances(voteID) = ElementMetric::evalSimplePatchDistance(distanceVec);
	}

	// compute distance sigma
	if (numVotes) {
		vector<double> vDist(voteDistances.data(), voteDistances.data() + numVotes);
		int n = (int)(numVotes* StyleSimilarityConfig::mOptimization_DistanceSigmaPercentile );
		nth_element(vDist.begin(), vDist.begin() + n, vDist.end());
		mpData->mDistanceSigma = vDist[n] * sigma;
	}

	// compute vote weight
	Eigen::VectorXd voteWeights;
	if (numVotes) {

		voteWeights = (-voteDistances.array() / mpData->mDistanceSigma).exp();

		for (int voteID = 0; voteID < numVotes; voteID++) {
			double area = (double)(mpData->mSourceSegmentsIndices[mTransformationVotes[voteID].patchSourceID].size());
			voteWeights(voteID) *= area;
		}
	}

	voteWeights /= voteWeights.sum(); // normalize	

	// compute mesh scale (for translation component normalization)

	double meshScale;
	if (true) {
		vec3 meshBBMin, meshBBMax;
		if (!MeshUtil::computeAABB(mpData->mTargetMesh, meshBBMin, meshBBMax)) return false;
		meshScale = (double)(meshBBMax - meshBBMin).length();
	}

	// build data matrix for mean-shift clustering
	vector<vector<double>> clusterData(numVotes); // high-dimension data : number of points (transformations)
	vector<double> clusterWeight(numVotes); // weight of point : number of points (transformations)	
	mTSpaceVotes.resize(numVotes, 10);
#pragma omp parallel for
	for (int voteID = 0; voteID < numVotes; voteID++) {

		auto &vote = mTransformationVotes[voteID];
		vector<double> trs;
		ElementUtil::convertAffineToTRS(vote.transformation, trs);
		auto &data = clusterData[voteID];
		auto &weight = clusterWeight[voteID];

		data.resize(9);
		for (int j = 0; j < 3; j++) data[j] = trs[j] / meshScale; // normalize by mesh size
		for (int j = 3; j < 6; j++) data[j] = trs[j+1]; // imaginary part of quaternion
		for (int j = 6; j < 9; j++) data[j] = trs[j+1];
		weight = voteWeights(voteID);
		//weight = vote.weight;

		for (int j = 0; j < 9; j++) mTSpaceVotes(voteID, j) = data[j];
		mTSpaceVotes(voteID, 9) = weight;
	}

	// perform mean shift clustering (using my own implementation)
	vector<int> clusterMode, clusterIndex;
	ClusterMeanShift cms(clusterData, clusterWeight);
	if (!cms.setBandwidth(-1)) return false; // adaptive bandwidth
	if (!cms.runClustering(clusterMode, clusterIndex)) return false;
	int numModes = (int)clusterMode.size();

	// push back modes
	mpData->mTransformationModes.resize(numModes);
	mTransformationModeFromVote.resize(numModes);
	for (int modeID = 0; modeID < numModes; modeID++) {		
		mpData->mTransformationModes[modeID] = mTransformationVotes[clusterMode[modeID]].transformation;
		mTransformationModeFromVote[modeID] = clusterMode[modeID];
	}
	mTransformationVoteFromMode = clusterIndex;

	return true;
}

bool ElementVoting::adjustModes() {

	bool skipAdjusting = true;

	vector<Eigen::Affine3d> validModes;
	vector<double> validModesWeight; // only used for sorting

	// always use global alignment as one transformation

	if (true) {
		double weight = 0;

		if (!skipAdjusting) {
			double distance = ElementMetric::evalSimplePatchDistance(mGlobalDistance);
			weight = exp(-distance / mpData->mDistanceSigma);
		}

		validModes.push_back(mGlobalTransformation);
		validModesWeight.push_back(weight);
	}
	

	// adjust modes by better ICP

	//*
	Eigen::Matrix3Xd matSourcePosition, matSourceNormal, matTargetPosition, matTargetNormal;
	if (!SampleUtil::buildMatrices(mpData->mSourceSamples, matSourcePosition, matSourceNormal)) return false;
	if (!SampleUtil::buildMatrices(mpData->mTargetSamples, matTargetPosition, matTargetNormal)) return false;
	//*/

	cout << "Adjusting modes" << endl;

	int numModes = (int)mpData->mTransformationModes.size();
	
	if (!skipAdjusting) {

//#pragma omp parallel for
		for (int modeID = 0; modeID < numModes; modeID++) {

			Eigen::Affine3d transformation = mpData->mTransformationModes[modeID];

			int voteID = mTransformationModeFromVote[modeID];
			auto &vote = mTransformationVotes[voteID];

			//// adjust patch only
			//Eigen::Matrix3Xd matSP, matSN, matTP, matTN;
			//if (!SampleUtil::buildMatrices(mpData->mSourceSegments[vote.patchSourceID], matSP, matSN)) error("build matrices");
			//if (!SampleUtil::buildMatrices(mpData->mTargetSegments[vote.patchTargetID], matTP, matTN)) error("build matrices");
			//if (!MatchSimpleICP::run(20, // UNDONE: param ICP iteration for adjusting modes
			//	matSP, matSN, matTP, matTN,
			//	transformation, true)) error("run ICP");

			// adjust entire shape
			Eigen::Matrix3Xd matXSP = transformation * matSourcePosition;
			Eigen::Matrix3Xd matXSN = transformation.rotation() * matSourceNormal;
			Eigen::Matrix3Xd matTP = matTargetPosition;
			Eigen::Matrix3Xd matTN = matTargetNormal;
			if (!MatchSimpleICP::prealign(matXSP, matXSN, matTP, matTN)) error("prealign");
			Eigen::Affine3d newTransformation = Eigen::Affine3d::Identity();
			if (!MatchSimpleICP::run(20, // UNDONE: param ICP iteration for adjusting modes
				matXSP, matXSN, matTargetPosition, matTargetNormal,
				newTransformation, true)) error("run ICP");
			transformation = newTransformation * transformation;

			mpData->mTransformationModes[modeID] = transformation;

			cout << "\rAdjusted modes " << (modeID + 1) << " / " << numModes << "          ";
		}
		cout << endl;
	}

	cout << "Computing mode weights" << endl;

	vector<double> allModesWeight(numModes, 1);
#pragma omp parallel for
	for (int modeID = 0; modeID < numModes; modeID++) {

		Eigen::Affine3d &transformation = mpData->mTransformationModes[modeID];
		int voteID = mTransformationModeFromVote[modeID];
		auto &vote = mTransformationVotes[voteID];

		Eigen::VectorXd simpleDistances;
		if (!ElementMetric::computeSimplePatchDistance(
			mpData->mSourceSamples, mpData->mTargetSamples,
			mpData->mSourceSegmentsIndices[vote.patchSourceID],
			mpData->mTargetSegmentsIndices[vote.patchTargetID],
			mpData->mpSourceFeatures, mpData->mpTargetFeatures,
			transformation, simpleDistances)) error("compute SPD");

		double distance = ElementMetric::evalSimplePatchDistance(simpleDistances);
		double area = (mpData->mSourceSegmentsIndices[vote.patchSourceID].size()
			+ mpData->mTargetSegmentsIndices[vote.patchTargetID].size()) * 0.5;
		double weight = area * exp(-distance / mpData->mDistanceSigma);

		allModesWeight[modeID] = weight;
	}

	// prune invalid modes & merge similar modes

	cout << "Pruning invalid modes" << endl;

	const double simThresholdT = mpData->mTargetSamples.radius * 2.0;
	const double simThresholdR = cml::rad(5.0);
	const double simThresholdS = 1.1;

	vector<int> modeIDMap(numModes, -1);
	if (true) {

		vector<int> sortIndex(numModes);
		for (int k = 0; k < numModes; k++) sortIndex[k] = k;
		sort(sortIndex.begin(), sortIndex.end(),
			[&allModesWeight](int i1, int i2) {return allModesWeight[i1] > allModesWeight[i2]; });

		for (int sortedID = 0; sortedID < numModes; sortedID++) {
			int modeID = sortIndex[sortedID];

			auto &transformation = mpData->mTransformationModes[modeID];
			vector<double> mode; // TRS
			if (!ElementUtil::convertAffineToTRS(transformation, mode)) return false;

			Eigen::Vector3d    modeT(mode[0], mode[1], mode[2]);
			Eigen::Quaterniond modeR(mode[3], mode[4], mode[5], mode[6]);
			Eigen::Vector3d    modeS(mode[7], mode[8], mode[9]);

			// prune invalid transformation
			bool flag = true;
			for (int j = 0; j < 3; j++) { // weird scaling
				double s = fabs(modeS[j]);
				if (s > 5.0 || s < 0.2) flag = false; // UNDONE: param pruning scale factor
			}
			if (flag) {
				for (auto &prevTransformation : validModes) { // similar to previous mode
					vector<double> prevMode; // TRS
					if (!ElementUtil::convertAffineToTRS(prevTransformation, prevMode)) return false;
					Eigen::Vector3d    prevT(prevMode[0], prevMode[1], prevMode[2]);
					Eigen::Quaterniond prevR(prevMode[3], prevMode[4], prevMode[5], prevMode[6]);
					Eigen::Vector3d    prevS(prevMode[7], prevMode[8], prevMode[9]);
					if ((modeT - prevT).norm() < simThresholdT &&
						modeR.angularDistance(prevR) < simThresholdR &&
						(modeS.array() / prevS.array()).maxCoeff() < simThresholdS &&
						(modeS.array() / prevS.array()).minCoeff() > 1.0 / simThresholdS)
					{
						flag = false;
						break;
					}
				}
			}
			if (flag) {
				modeIDMap[modeID] = (int)validModes.size();
				validModes.push_back(transformation);
				validModesWeight.push_back(allModesWeight[modeID]);
			}
		}
	}

	// export valid modes sorted by error

	int numValidModes = (int)validModes.size();
	if (numValidModes != (int)validModesWeight.size()) {
		cout << "Error: inconsistent size between validModes and validModesWeight" << endl;
		return false;
	}
	if (true) {
		vector<int> sortIndex(numValidModes);
		for (int k = 0; k < numValidModes; k++) sortIndex[k] = k;
		sort(sortIndex.begin() + 1, sortIndex.end(), // don't sort the first mode (global alignment)
			[&validModesWeight](int i1, int i2) {return validModesWeight[i1] > validModesWeight[i2]; });

		mpData->mTransformationModes.clear();
		mpData->mTransformationModes.reserve(numValidModes);
		vector<int> validIDMap(numValidModes, -1);
		for (int idx : sortIndex) {
			validIDMap[idx] = (int)mpData->mTransformationModes.size();
			mpData->mTransformationModes.push_back(validModes[idx]);
		}

		cout << "Valid modes: " << numValidModes << endl;

		// update vote from mode
		for (int &modeID : mTransformationVoteFromMode) {
			if (modeID >= 0) {
				int validID = modeIDMap[modeID];
				if (validID >= 0) {
					int finalID = validIDMap[validID];
					if (finalID >= 0) modeID = finalID;
					else modeID = -1;
				}
				else modeID = -1;
			}
		}

	}

	// export TSpace modes data

	double meshScale;
	if (true) {
		vec3 meshBBMin, meshBBMax;
		if (!MeshUtil::computeAABB(mpData->mTargetMesh, meshBBMin, meshBBMax)) return false;
		meshScale = (double)(meshBBMax - meshBBMin).length();
	}

	numModes = (int)mpData->mTransformationModes.size();
	mTSpaceModes.resize(numModes, 9);
	for (int modeID = 0; modeID < numModes; modeID++) {
		Eigen::Affine3d mode = mpData->mTransformationModes[modeID];
		vector<double> trs;
		if (!ElementUtil::convertAffineToTRS(mode, trs)) return false;
		for (int j = 0; j < 3; j++) mTSpaceModes(modeID, j) = trs[j] / meshScale; // normalize by mesh size
		for (int j = 3; j < 6; j++) mTSpaceModes(modeID, j) = trs[j + 1]; // imaginary part of quaternion
		for (int j = 6; j < 9; j++) mTSpaceModes(modeID, j) = trs[j + 1];
	}

	return true;
}

bool ElementVoting::checkPatchPairs(
	double sourceVolume,
	double targetVolume,
	Eigen::Vector3d &sourceVariance,
	Eigen::Vector3d &targetVariance)
{
	// UNDONE: param for preliminary pruning
	double varianceFactor = 5.0;
	double volumeFactor = 100.0;

	if (sourceVolume > targetVolume * volumeFactor ||
		sourceVolume < targetVolume / volumeFactor) return false;

	for (int j = 0; j < 3; j++) {
		if (sourceVariance[j] > targetVariance[j] * varianceFactor ||
			sourceVariance[j] < targetVariance[j] / varianceFactor) return false;
	}

	return true;
}

bool ElementVoting::computePatchIntrinsics(
	Eigen::Matrix3Xd &inPatch,
	double &inRadius,
	double &outVolume,
	Eigen::Vector3d &outVariance)
{
	if (inPatch.cols() < 3) { // cannot apply SVD
		outVolume = 0;
		outVariance.setZero();
		return true;
	}

	// PCA (perform SVD rather than eigen solver for better numerical precision)
	Eigen::Matrix3Xd matO = inPatch.colwise() - inPatch.rowwise().mean(); // zero-mean points
	Eigen::JacobiSVD< Eigen::Matrix3Xd > svd(matO, Eigen::ComputeThinU);
	Eigen::Matrix3d matU = svd.matrixU();

	// compute OBB extent
	Eigen::Matrix3Xd matT = matU.transpose() * matO; // transform points to OBB local CS
	Eigen::Vector3d vecOBB = (matT.rowwise().maxCoeff() - matT.rowwise().minCoeff()).array().max(inRadius);

	// compute anisotropy (following APS paper)
	double s1 = svd.singularValues()[0];
	double s2 = svd.singularValues()[1];
	double s3 = svd.singularValues()[2];
	double cL = (s1 - s2) / (s1 + s2 + s3);     // linearity
	double cP = 2 * (s2 - s3) / (s1 + s2 + s3); // planarity
	double cS = 3 * s3 / (s1 + s2 + s3);        // sphericity

	outVolume = vecOBB[0] * vecOBB[1] * vecOBB[2];
	outVariance = Eigen::Vector3d(cL, cP, cS);

	return true;
}

bool ElementVoting::saveVotes(string fileName) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << mTransformationVotes.size() << endl;
	for (auto &vote : mTransformationVotes) {
		file << vote.patchSourceID << " " << vote.patchTargetID << " " << vote.weight << endl;
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 4; c++) {
				file << vote.transformation(r, c) << " ";
			}
			file << endl;
		}
	}
	file.close();

	return true;
}

bool ElementVoting::loadVotes(string fileName) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int numPairs;
	file >> numPairs;
	mTransformationVotes.clear();
	for (int pairID = 0; pairID < numPairs; pairID++) {
		TTransformVote vote;
		file >> vote.patchSourceID >> vote.patchTargetID >> vote.weight;
		file.ignore(); // proceed to next line
		bool valid = true;
		for (int r = 0; r < 3; r++) {
			string line;
			getline(file, line);
			if (!valid) continue;
			else if (line.find_first_of('#') != string::npos) {
				valid = false;
			} else {
				stringstream ss(line);
				for (int c = 0; c < 4; c++) {
					double v; ss >> v;
					vote.transformation(r, c) = v;
				}
			}
		}
		if(valid) mTransformationVotes.push_back(vote);
	}
	file.close();

	return true;
}

bool ElementVoting::saveVoteDistance(string fileName) {

	return ElementUtil::saveMatrixBinary(fileName, mTransformationVoteDistance);
}

bool ElementVoting::loadVoteDistance(string fileName) {

	return ElementUtil::loadMatrixBinary(fileName, mTransformationVoteDistance);
}

bool ElementVoting::saveGlobalAlignment(string fileName) {

	if (!mGlobalTransformation.matrix().allFinite()) {
		cout << "Error: NaN in global transformation" << endl;
		return false;
	}
	if (!mGlobalDistance.allFinite()) {
		cout << "Error: NaN in global distance" << endl;
		return false;
	}

	ofstream file(fileName);
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 4; c++) {
			file << mGlobalTransformation(r, c) << " ";
		}
		file << endl;
	}
	for (int d = 0; d < (int)mGlobalDistance.size(); d++) {
		file << mGlobalDistance(d) << " ";
	}
	file << endl;
	file.close();

	return true;
}

bool ElementVoting::loadGlobalAlignment(string fileName) {

	ifstream file(fileName);
	mGlobalTransformation.setIdentity();
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 4; c++) {
			double v; file >> v;
			mGlobalTransformation(r, c) = v;
		}
	}
	vector<double> dist;
	while (!file.eof()) {
		double v; file >> v;
		if (file.fail()) break;
		dist.push_back(v);
	}
	int dim = (int)dist.size();
	mGlobalDistance.resize(dim);
	for (int d = 0; d < dim; d++) {
		mGlobalDistance(d) = dist[d];
	}
	file.close();

	return true;
}

bool ElementVoting::saveVoteModes(string fileName) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << (int)mTransformationVoteFromMode.size() << " ";
	for (int modeID : mTransformationVoteFromMode) {
		file << modeID << " ";
	}
	file << endl;
	file.close();

	return true;
}


bool ElementVoting::loadVoteModes(string fileName) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int numVotes;
	file >> numVotes;
	mTransformationVoteFromMode.resize(numVotes);
	for (int voteID = 0; voteID < numVotes; voteID++) {
		int modeID; file >> modeID;
		mTransformationVoteFromMode[voteID] = modeID;
	}
	file.close();

	return true;
}

bool ElementVoting::saveModes(string fileName) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << mpData->mTransformationModes.size() << " " << mpData->mDistanceSigma << endl;
	for (auto &mode : mpData->mTransformationModes) {
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 4; c++) {
				file << mode(r, c) << " ";
			}
			file << endl;
		}
	}
	file.close();

	return true;
}

bool ElementVoting::loadModes(string fileName) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int numModes;
	double distanceSigma;
	file >> numModes;
	file >> distanceSigma;
	mpData->mDistanceSigma = distanceSigma;
	mpData->mTransformationModes.resize(numModes);
	for (int modeID = 0; modeID < numModes; modeID++) {
		auto &mode = mpData->mTransformationModes[modeID];
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 4; c++) {
				double v; file >> v;
				mode(r, c) = v;
			}
		}
	}
	file.close();

	return true;
}

bool ElementVoting::visualizeModes(string fileName) {

	// get bounding box
	vec3 bbScale;
	if (true) {
		vec3 bbMin, bbMax;
		if (!SampleUtil::computeAABB(mpData->mTargetSamples, bbMin, bbMax)) return false;
		bbScale = bbMax - bbMin;		
		bbScale *= 1.1f;
	}

	PlyExporter pe;
	for (int modeID = 0; modeID < (int)mpData->mTransformationModes.size(); modeID++) {

		Eigen::Affine3d transformation = mpData->mTransformationModes[modeID];

		vec3 vOffset = vec3(float(modeID % 4) * bbScale[0], 0.0f, -float(modeID / 4) * bbScale[2]);
		transformation.pretranslate(Eigen::Vector3d(vOffset[0], vOffset[1], vOffset[2]));
		matrix4ed matFinalE(transformation.matrix().data());
		matrix4f matFinal = matFinalE;

		if (!pe.addPoint(&mpData->mSourceSamples.positions, &mpData->mSourceSamples.normals, matFinal, vec3i(127, 255, 255))) return false;
		if (!pe.addPoint(&mpData->mTargetSamples.positions, &mpData->mTargetSamples.normals, vOffset, vec3i(127, 127, 127))) return false;
	}
	if (!pe.output(fileName)) return false;

	return true;
}

bool ElementVoting::exportTSpaceData(string voteFileName, string modeFileName) {

	if (!ElementUtil::saveMatrixASCII(voteFileName, mTSpaceVotes)) return false;
	if (!ElementUtil::saveMatrixASCII(modeFileName, mTSpaceModes)) return false;

	return true;
}

void ElementVoting::error(string s) {
	cout << "Error: " << s << endl;
}