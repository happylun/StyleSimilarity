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

#include "ElementMetric.h"

#include <fstream>

#include "Match/MatchSimpleICP.h"

#include "Mesh/MeshUtil.h"
#include "Sample/SampleUtil.h"
#include "Segment/SegmentUtil.h"
#include "Element/ElementUtil.h"

#include "Utility/PlyExporter.h"

#include "Data/StyleSimilarityConfig.h"

#define OUTPUT_PROGRESS

using namespace StyleSimilarity;

Eigen::VectorXd ElementMetric::mWeightsSimplePatchDistance;
Eigen::VectorXd ElementMetric::mWeightsFullPatchDistance;
Eigen::VectorXd ElementMetric::mWeightsFullShapeDistance;
Eigen::VectorXd ElementMetric::mWeightsFullSaliency;

Eigen::VectorXd ElementMetric::mScaleSimplePatchDistance;
Eigen::VectorXd ElementMetric::mScaleFullPatchDistance;
Eigen::VectorXd ElementMetric::mScaleFullShapeDistance;
Eigen::VectorXd ElementMetric::mScaleFullSaliency;

bool ElementMetric::computePointSaliency(
	TTriangleMesh &inMesh,
	TSampleSet &inSample,
	FeatureAsset &inFeature,
	Eigen::MatrixXd &outSaliency)
{
	vec3 bbMin, bbMax;
	if (!MeshUtil::computeAABB(inMesh, bbMin, bbMax)) return false;

	int numPoints = inSample.amount;
	vector<vector<double>> tmpSaliency(numPoints);

#pragma omp parallel for
	for (int pointID = 0; pointID < numPoints; pointID++) {
		
		vec3 v = inSample.positions[pointID];
		double saliencyHeight = (double)((v[1] - bbMin[1]) / (bbMax[1] - bbMin[1]));
		double saliencyRadius = (double)((v * 2 - (bbMin + bbMax)).length_squared() / (bbMax - bbMin).length_squared());

		double saliencyGeodesic = inFeature.mGeodesic[pointID];
		double saliencyAO = inFeature.mAO[pointID];

		auto &curv = inFeature.mCurvature[pointID];
		vector<double> saliencyCurvature;
		if (!FeatureCurvature::getSaliencyMetrics(curv, saliencyCurvature)) error("get saliency metrics");

		vector<double> saliencyTal;
		for (int j = 0; j < 3; j++) {
			saliencyTal.push_back(inFeature.mTalFPFH[pointID][j]);
			saliencyTal.push_back(inFeature.mTalSI[pointID][j]);
			saliencyTal.push_back(inFeature.mTalSC[pointID][j]);
		}

		vector<double> &allSaliencies = tmpSaliency[pointID];
		allSaliencies.clear();
		allSaliencies.push_back(saliencyHeight);
		allSaliencies.push_back(saliencyRadius);
		allSaliencies.push_back(saliencyGeodesic);
		allSaliencies.push_back(saliencyAO);
		allSaliencies.insert(allSaliencies.end(), saliencyCurvature.begin(), saliencyCurvature.end());
		allSaliencies.insert(allSaliencies.end(), saliencyTal.begin(), saliencyTal.end());
	}

	int dim = (int)tmpSaliency[0].size();
	outSaliency.resize(numPoints, dim);
	for (int r = 0; r < numPoints; r++) {
		for (int c = 0; c < dim; c++) {
			outSaliency(r, c) = tmpSaliency[r][c];
		}
	}

	if (!outSaliency.allFinite()) {
		cout << "Error: NaN point saliency" << endl;
		return false;
	}

	return true;
}


bool ElementMetric::computeVerySimplePatchDistance(
	TSampleSet &inSourceShape,
	TSampleSet &inTargetShape,
	vector<int> &inSourcePatch,
	vector<int> &inTargetPatch,
	Eigen::Affine3d &inTransform,
	Eigen::VectorXd &outDistance)
{
	// prepare data
	TPointSet sourcePointSet, targetPointSet;
	if (!SegmentUtil::extractPointSet(inSourceShape, inSourcePatch, sourcePointSet)) return false;
	if (!SegmentUtil::extractPointSet(inTargetShape, inTargetPatch, targetPointSet)) return false;

	Eigen::Matrix3Xd matSP, matSN;
	Eigen::Matrix3Xd matTP, matTN;
	if (!SampleUtil::buildMatrices(sourcePointSet, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(targetPointSet, matTP, matTN)) return false;
	Eigen::Matrix3Xd matXSP = inTransform * matSP;
	Eigen::Matrix3Xd matXSN = inTransform.rotation() * matSN;

	outDistance.resize(mWeightsSimplePatchDistance.size());
	outDistance.setZero();

	// point cloud alignment distance
	{
		double distP, distN;
		if (!MatchSimpleICP::distance(
			matXSP, matXSN, matTP, matTN,
			distP, distN)) return false;
		distP = distP / inTargetShape.radius; // normalize distance

		outDistance(0) = distP;
		outDistance(1) = distN;
	}

	if (!outDistance.allFinite()) {
		cout << "Error: NaN in Very Simple Patch Distance" << endl;
		return false;
	}

	return true;
}

bool ElementMetric::computeSimplePatchDistance(
	TSampleSet &inSourceShape,
	TSampleSet &inTargetShape,
	vector<int> &inSourcePatch,
	vector<int> &inTargetPatch,
	FeatureAsset *inSourceFeatures,
	FeatureAsset *inTargetFeatures,
	Eigen::Affine3d &inTransform,
	Eigen::VectorXd &outDistance)
{
	if (inSourceFeatures == 0 || inTargetFeatures == 0) {
		return computeVerySimplePatchDistance(
			inSourceShape, inTargetShape,
			inSourcePatch, inTargetPatch,
			inTransform, outDistance);
	}

	// prepare data
	TPointSet sourcePointSet, targetPointSet;
	if (!SegmentUtil::extractPointSet(inSourceShape, inSourcePatch, sourcePointSet)) return false;
	if (!SegmentUtil::extractPointSet(inTargetShape, inTargetPatch, targetPointSet)) return false;

	Eigen::Matrix3Xd matSP, matSN;
	Eigen::Matrix3Xd matTP, matTN;
	if (!SampleUtil::buildMatrices(sourcePointSet, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(targetPointSet, matTP, matTN)) return false;
	Eigen::Matrix3Xd matXSP = inTransform * matSP;
	Eigen::Matrix3Xd matXSN = inTransform.rotation() * matSN;

	// point cloud alignment distance
	vector<double> alignDistance;
	{
		double distP, distN;
		if (!MatchSimpleICP::distance(
			matXSP, matXSN, matTP, matTN,
			distP, distN)) return false;
		distP = distP / inTargetShape.radius; // normalize distance

		alignDistance.clear();
		alignDistance.push_back(distP);
		alignDistance.push_back(distN);
	}

	// curvature distances
	vector<double> curvatureDistance;
	{
		vector<FeatureCurvature::TCurvature> srcCurv;
		vector<FeatureCurvature::TCurvature> tgtCurv;
		for (int id : inSourcePatch) srcCurv.push_back(inSourceFeatures->mCurvature[id]);
		for (int id : inTargetPatch) tgtCurv.push_back(inTargetFeatures->mCurvature[id]);
		if (!FeatureCurvature::compareFeatures(srcCurv, tgtCurv, curvatureDistance)) return false;
	}

	// SDF distance
	vector<double> sdfDistance;
	{
		vector<double> srcSDF;
		vector<double> tgtSDF;
		for (int id : inSourcePatch) srcSDF.push_back(inSourceFeatures->mSDF[id]);
		for (int id : inTargetPatch) tgtSDF.push_back(inTargetFeatures->mSDF[id]);
		if (!FeatureSDF::compareFeatures(srcSDF, tgtSDF, sdfDistance)) return false;
	}

	vector<double> allDistances;
	allDistances.clear();
	allDistances.insert(allDistances.end(), alignDistance.begin(), alignDistance.end());
	allDistances.insert(allDistances.end(), curvatureDistance.begin(), curvatureDistance.end());
	allDistances.insert(allDistances.end(), sdfDistance.begin(), sdfDistance.end());

	outDistance.resize(allDistances.size());
	for (int k = 0; k < (int)allDistances.size(); k++) {
		outDistance(k) = allDistances[k];
	}

	if (!outDistance.allFinite()) {
		cout << "Error: NaN in Simple Patch Distance" << endl;
		return false;
	}

	return true;
}


bool ElementMetric::computeFullPatchDistance(
	TSampleSet &inSourceShape,
	TSampleSet &inTargetShape,
	vector<int> &inSourcePatch,
	vector<int> &inTargetPatch,
	FeatureAsset &inSourceFeatures,
	FeatureAsset &inTargetFeatures,
	Eigen::Affine3d &inTransform,
	Eigen::VectorXd &outDistance)
{

	// prepare data
	TSampleSet sourcePointSet, targetPointSet;
	if (!SegmentUtil::extractSampleSet(inSourceShape, inSourcePatch, sourcePointSet)) return false;
	if (!SegmentUtil::extractSampleSet(inTargetShape, inTargetPatch, targetPointSet)) return false;
	Eigen::Matrix3Xd matSP, matSN;
	Eigen::Matrix3Xd matTP, matTN;
	if (!SampleUtil::buildMatrices(sourcePointSet, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(targetPointSet, matTP, matTN)) return false;
	auto inRotation = inTransform.rotation();
	Eigen::Matrix3Xd matXSP = inTransform * matSP;
	Eigen::Matrix3Xd matXSN = inRotation * matSN;

	// point cloud alignment distance
	vector<double> alignDistance;
	{
		double distP, distN;
		if (!MatchSimpleICP::distance(
			matXSP, matXSN, matTP, matTN,
			distP, distN)) return false;
		distP = distP / inTargetShape.radius; // normalize distance

		alignDistance.clear();
		alignDistance.push_back(distP);
		alignDistance.push_back(distN);
	}

	// curves alignment distance
	vector<double> curveDistance;
	{
		curveDistance.clear();
		for (int curveID = 0; curveID < CurveRidgeValley::CURVE_TYPES; curveID++) {

			// build matrix
			Eigen::VectorXi srcIndices, tgtIndices;
			if (!SampleUtil::findNearestNeighbors(inSourceFeatures.mCurveTree[curveID], matSP, srcIndices)) return false;
			if (!SampleUtil::findNearestNeighbors(inTargetFeatures.mCurveTree[curveID], matTP, tgtIndices)) return false;
			set<int> srcSet(srcIndices.data(), srcIndices.data() + srcIndices.size());
			set<int> tgtSet(tgtIndices.data(), tgtIndices.data() + tgtIndices.size());
			Eigen::Matrix3Xd matCSP(3, srcSet.size());
			Eigen::Matrix3Xd matCSN(3, srcSet.size());
			Eigen::Matrix3Xd matCTP(3, tgtSet.size());
			Eigen::Matrix3Xd matCTN(3, tgtSet.size());
			if (srcSet.size() <= 1) { // no matching curve points
				matCSP = matSP.leftCols(1);
				matCSN = matSN.leftCols(1);
			} else {
				int count = 0;
				for (int id : srcSet) {
					matCSP.col(count) = Eigen::Vector3d(vec3d(inSourceFeatures.mCurve[curveID].positions[id]).data());
					matCSN.col(count) = Eigen::Vector3d(vec3d(inSourceFeatures.mCurve[curveID].normals[id]).data());
					count++;
				}
			}
			if (tgtSet.size() <= 1) { // no matching curve points
				matCTP = matTP.leftCols(1);
				matCTN = matTN.leftCols(1);
			} else {
				int count = 0;
				for (int id : tgtSet) {
					matCTP.col(count) = Eigen::Vector3d(vec3d(inTargetFeatures.mCurve[curveID].positions[id]).data());
					matCTN.col(count) = Eigen::Vector3d(vec3d(inTargetFeatures.mCurve[curveID].normals[id]).data());
					count++;
				}
			}
			Eigen::Matrix3Xd matCXP = inTransform * matCSP;
			Eigen::Matrix3Xd matCXN = inRotation * matCSN;

			double distP, distN;
			if (!MatchSimpleICP::distance(matCXP, matCXN, matCTP, matCTN, distP, distN)) return false;
			distP = distP / inTargetShape.radius; // normalize distance

			// HACK: NaN case
			if (!matCXP.allFinite() || !matCXN.allFinite() || !matCTP.allFinite() || !matCTN.allFinite()) {
				distP = 5.0;
				distN = 2.0;
			}

			curveDistance.push_back(distP);
			curveDistance.push_back(distN);
		}
	}

	// scale distance
	vector<double> scaleDistances;
	{
		vector<double> trs;
		ElementUtil::convertAffineToTRS(inTransform, trs);
		double sX = max(fabs(trs[7]), 1 / max(0.01, fabs(trs[7]))) - 1;
		double sY = max(fabs(trs[8]), 1 / max(0.01, fabs(trs[8]))) - 1;
		double sZ = max(fabs(trs[9]), 1 / max(0.01, fabs(trs[9]))) - 1;
		scaleDistances.push_back(sX);
		scaleDistances.push_back(sY);
		scaleDistances.push_back(sZ);
	}

	// SD distance
	vector<double> sdDistance;
	{
		vector<double> srcSD;
		vector<double> tgtSD;
		FeatureShapeDistributions sdS(&sourcePointSet, &srcSD);
		if (!sdS.calculate()) return false;
		FeatureShapeDistributions sdT(&targetPointSet, &tgtSD);
		if (!sdT.calculate()) return false;
		if (!FeatureShapeDistributions::compareFeatures(srcSD, tgtSD, sdDistance)) return false;
	}

	// curvature distances
	vector<double> curvatureDistance;
	{
		vector<FeatureCurvature::TCurvature> srcCurv;
		vector<FeatureCurvature::TCurvature> tgtCurv;
		for (int id : inSourcePatch) srcCurv.push_back(inSourceFeatures.mCurvature[id]);
		for (int id : inTargetPatch) tgtCurv.push_back(inTargetFeatures.mCurvature[id]);
		if (!FeatureCurvature::compareFeatures(srcCurv, tgtCurv, curvatureDistance)) return false;
	}

	// SDF distance
	vector<double> sdfDistance;
	{
		vector<double> srcSDF;
		vector<double> tgtSDF;
		for (int id : inSourcePatch) srcSDF.push_back(inSourceFeatures.mSDF[id]);
		for (int id : inTargetPatch) tgtSDF.push_back(inTargetFeatures.mSDF[id]);
		if (!FeatureSDF::compareFeatures(srcSDF, tgtSDF, sdfDistance)) return false;
	}

	// LFD distance
	vector<double> lfdDistance;
#pragma omp critical
	{
		if (!FeatureLFD::init()) error("init LFD"); // don't release render context
		vector<double> srcLFD;
		vector<double> tgtLFD;
		FeatureLFD lfdS(&sourcePointSet, &srcLFD);
		if (!lfdS.calculate()) error("source LFD");
		FeatureLFD lfdT(&targetPointSet, &tgtLFD);
		if (!lfdT.calculate()) error("target LFD");
		if (!FeatureLFD::compareFeatures(srcLFD, tgtLFD, lfdDistance)) error("compare LFD");
	}

	vector<double> allDistances;
	allDistances.clear();
	allDistances.insert(allDistances.end(), alignDistance.begin(), alignDistance.end());
	allDistances.insert(allDistances.end(), curveDistance.begin(), curveDistance.end());
	allDistances.insert(allDistances.end(), scaleDistances.begin(), scaleDistances.end());
	allDistances.insert(allDistances.end(), sdDistance.begin(), sdDistance.end());
	allDistances.insert(allDistances.end(), curvatureDistance.begin(), curvatureDistance.end());
	allDistances.insert(allDistances.end(), sdfDistance.begin(), sdfDistance.end());
	allDistances.insert(allDistances.end(), lfdDistance.begin(), lfdDistance.end());

	outDistance.resize(allDistances.size());
	for (int k = 0; k < (int)allDistances.size(); k++) {
		outDistance(k) = allDistances[k];
	}

	if (!outDistance.allFinite()) {
		cout << "Error: NaN in Full Patch Distance" << endl;
		return false;
	}

	return true;
}

bool ElementMetric::computeFullShapeDistance(
	TSampleSet &inSourceShape,
	TSampleSet &inTargetShape,
	FeatureAsset &inSourceFeatures,
	FeatureAsset &inTargetFeatures,
	Eigen::Affine3d &inTransform,
	Eigen::VectorXd &outDistance)
{

	// prepare data
	Eigen::Matrix3Xd matSP, matSN;
	Eigen::Matrix3Xd matTP, matTN;
	if (!SampleUtil::buildMatrices(inSourceShape, matSP, matSN)) return false;
	if (!SampleUtil::buildMatrices(inTargetShape, matTP, matTN)) return false;
	auto inRotation = inTransform.rotation();
	Eigen::Matrix3Xd matXSP = inTransform * matSP;
	Eigen::Matrix3Xd matXSN = inRotation * matSN;

	// point cloud alignment distance
	vector<double> alignDistance;
	{
		double distP, distN;
		if (!MatchSimpleICP::distance(
			matXSP, matXSN, matTP, matTN,
			distP, distN)) return false;
		distP = distP / inTargetShape.radius; // normalize distance

		alignDistance.clear();
		alignDistance.push_back(distP);
		alignDistance.push_back(distN);
	}

	// curves alignment distance
	vector<double> curveDistance;
	{
		curveDistance.clear();
		for (int curveID = 0; curveID < CurveRidgeValley::CURVE_TYPES; curveID++) {

			// build matrix
			Eigen::Matrix3Xd matCSP, matCSN;
			Eigen::Matrix3Xd matCTP, matCTN;
			if (!SampleUtil::buildMatrices(inSourceFeatures.mCurve[curveID], matCSP, matCSN)) return false;
			if (!SampleUtil::buildMatrices(inTargetFeatures.mCurve[curveID], matCTP, matCTN)) return false;
			if (matCSP.cols() == 0) {
				matCSP = matSP.leftCols(1);
				matCSN = matSN.leftCols(1);
			}
			if (matCTP.cols() == 0) {
				matCTP = matTP.leftCols(1);
				matCTN = matTN.leftCols(1);
			}
			Eigen::Matrix3Xd matCXP = inTransform * matCSP;
			Eigen::Matrix3Xd matCXN = inRotation * matCSN;

			double distP, distN;
			if (!MatchSimpleICP::distance(matCXP, matCXN, matCTP, matCTN, distP, distN)) return false;
			distP = distP / inTargetShape.radius; // normalize distance

			// HACK: NaN case
			if (!matCXP.allFinite() || !matCXN.allFinite() || !matCTP.allFinite() || !matCTN.allFinite()) {
				distP = 5.0;
				distN = 2.0;
			}

			curveDistance.push_back(distP);
			curveDistance.push_back(distN);
		}
	}

	// scale distance
	vector<double> scaleDistance;
	{
		vector<double> trs;
		ElementUtil::convertAffineToTRS(inTransform, trs);
		double sX = max(fabs(trs[7]), 1 / max(0.01, fabs(trs[7]))) - 1;
		double sY = max(fabs(trs[8]), 1 / max(0.01, fabs(trs[8]))) - 1;
		double sZ = max(fabs(trs[9]), 1 / max(0.01, fabs(trs[9]))) - 1;
		scaleDistance.push_back(sX);
		scaleDistance.push_back(sY);
		scaleDistance.push_back(sZ);
	}

	// SD distance
	vector<double> sdDistance;
	{
		if (!FeatureShapeDistributions::compareFeatures(
			inSourceFeatures.mSD, inTargetFeatures.mSD,
			sdDistance)) return false;
	}

	// curvature distances
	vector<double> curvatureDistance;
	{
		if (!FeatureCurvature::compareFeatures(
			inSourceFeatures.mCurvature, inTargetFeatures.mCurvature,
			curvatureDistance)) return false;
	}


	// SDF distance
	vector<double> sdfDistance;
	{
		if (!FeatureSDF::compareFeatures(
			inSourceFeatures.mSDF, inTargetFeatures.mSDF,
			sdfDistance)) return false;
	}

	// LFD distance
	vector<double> lfdDistance;
	{
		if (!FeatureLFD::compareFeatures(
			inSourceFeatures.mLFD, inTargetFeatures.mLFD,
			lfdDistance)) return false;
	}

	/*
	// NOT USED
	// arrangement distance
	double arrDistance;
	{
		if (!FeatureArrangement::compareFeatures(
			inSourceFeatures.mArrangement, inTargetFeatures.mArrangement,
			arrDistance)) return false;
	}
	*/
	vector<double> allDistances;
	allDistances.clear();
	allDistances.insert(allDistances.end(), alignDistance.begin(), alignDistance.end());
	allDistances.insert(allDistances.end(), curveDistance.begin(), curveDistance.end());
	allDistances.insert(allDistances.end(), scaleDistance.begin(), scaleDistance.end());
	allDistances.insert(allDistances.end(), sdDistance.begin(), sdDistance.end());
	allDistances.insert(allDistances.end(), curvatureDistance.begin(), curvatureDistance.end());
	allDistances.insert(allDistances.end(), sdfDistance.begin(), sdfDistance.end());
	allDistances.insert(allDistances.end(), lfdDistance.begin(), lfdDistance.end());
	//allDistances.push_back(arrDistance); // NOT USED

	outDistance.resize(allDistances.size());
	for (int k = 0; k < (int)allDistances.size(); k++) {
		outDistance(k) = allDistances[k];
	}

	if (!outDistance.allFinite()) {
		cout << "Error: NaN in Full Shape Distance" << endl;
		return false;
	}

	return true;
}

bool ElementMetric::loadWeightsVector(string fileName, Eigen::VectorXd &weights) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	vector<double> weightList;
	while (!file.eof()) {
		double w; file >> w;
		if (file.fail()) break;
		weightList.push_back(w);
	}
	file.close();
	weights.resize(weightList.size());
	for (int k = 0; k < (int)weightList.size(); k++) weights(k) = weightList[k];

	return true;
}

void ElementMetric::error(string s) {
	cout << "Error: " << s << endl;
}