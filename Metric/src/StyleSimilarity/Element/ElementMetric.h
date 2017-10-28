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

#pragma once

#include <set>

#include "Data/StyleSimilarityTypes.h"
#include "Data/StyleSimilarityData.h"

#include "Feature/FeatureAsset.h"

using namespace std;

namespace StyleSimilarity {

	class ElementMetric {

	private:

		ElementMetric() {}
		~ElementMetric() {}

	public:

		static bool computePointSaliency(
			TTriangleMesh &inMesh,
			TSampleSet &inSample,
			FeatureAsset &inFeature,
			Eigen::MatrixXd &outSaliency);

		static bool computeVerySimplePatchDistance(
			TSampleSet &inSourceShape,
			TSampleSet &inTargetShape,
			vector<int> &inSourcePatch,
			vector<int> &inTargetPatch,
			Eigen::Affine3d &inTransform,
			Eigen::VectorXd &outDistance);

		static bool computeSimplePatchDistance(
			TSampleSet &inSourceShape,
			TSampleSet &inTargetShape,
			vector<int> &inSourcePatch,
			vector<int> &inTargetPatch,
			FeatureAsset *inSourceFeatures,
			FeatureAsset *inTargetFeatures,
			Eigen::Affine3d &inTransform,
			Eigen::VectorXd &outDistance);

		static bool computeFullPatchDistance(
			TSampleSet &inSourceShape,
			TSampleSet &inTargetShape,
			vector<int> &inSourcePatch,
			vector<int> &inTargetPatch,
			FeatureAsset &inSourceFeatures,
			FeatureAsset &inTargetFeatures,
			Eigen::Affine3d &inTransform,
			Eigen::VectorXd &outDistance);

		static bool computeFullShapeDistance(
			TSampleSet &inSourceShape,
			TSampleSet &inTargetShape,
			FeatureAsset &inSourceFeatures,
			FeatureAsset &inTargetFeatures,
			Eigen::Affine3d &inTransform,
			Eigen::VectorXd &outDistance);

		inline static bool loadWeightsSimplePatchDistance(string fileName) {
			return loadWeightsVector(fileName, mWeightsSimplePatchDistance);
		}
		inline static bool loadWeightsFullPatchDistance(string fileName) {
			return loadWeightsVector(fileName, mWeightsFullPatchDistance);
		}
		inline static bool loadWeightsFullShapeDistance(string fileName) {
			return loadWeightsVector(fileName, mWeightsFullShapeDistance);
		}
		inline static bool loadWeightsFullSaliency(string fileName) {
			return loadWeightsVector(fileName, mWeightsFullSaliency);
		}

		inline static bool loadScaleSimplePatchDistance(string fileName) {
			return loadWeightsVector(fileName, mScaleSimplePatchDistance);
		}
		inline static bool loadScaleFullPatchDistance(string fileName) {
			return loadWeightsVector(fileName, mScaleFullPatchDistance);
		}
		inline static bool loadScaleFullShapeDistance(string fileName) {
			return loadWeightsVector(fileName, mScaleFullShapeDistance);
		}
		inline static bool loadScaleFullSaliency(string fileName) {
			return loadWeightsVector(fileName, mScaleFullSaliency);
		}

		inline static double evalSimplePatchDistance(Eigen::VectorXd &distanceVec) {
			Eigen::VectorXd normalizedVec = distanceVec.cwiseQuotient(ElementMetric::mScaleSimplePatchDistance);
			//normalizedVec = normalizedVec.cwiseMin(1.0);
			return normalizedVec.dot(ElementMetric::mWeightsSimplePatchDistance);
		}

		inline static double evalFullPatchDistance(Eigen::VectorXd &distanceVec) {
			Eigen::VectorXd normalizedVec = distanceVec.cwiseQuotient(ElementMetric::mScaleFullPatchDistance);
			normalizedVec = normalizedVec.cwiseMin(1.0);
			return normalizedVec.dot(ElementMetric::mWeightsFullPatchDistance);
		}

		inline static double evalFullShapeDistance(Eigen::VectorXd &distanceVec) {
			Eigen::VectorXd normalizedVec = distanceVec.cwiseQuotient(ElementMetric::mScaleFullShapeDistance);
			normalizedVec = normalizedVec.cwiseMin(1.0);
			return normalizedVec.dot(ElementMetric::mWeightsFullShapeDistance);
		}

		inline static double evalFullSaliency(Eigen::VectorXd &saliencyVec) {
			Eigen::VectorXd normalizedVec = saliencyVec.cwiseQuotient(ElementMetric::mScaleFullSaliency);
			normalizedVec = normalizedVec.cwiseMin(1.0);
			Eigen::VectorXd homoVec(normalizedVec.size() + 1);
			homoVec << normalizedVec, 1.0;
			double s = homoVec.dot(ElementMetric::mWeightsFullSaliency);
			return 1.0 / (1.0 + exp(-s));
		}

	private:

		static bool loadWeightsVector(string fileName, Eigen::VectorXd &weights);

	public:

		static Eigen::VectorXd mWeightsSimplePatchDistance;
		static Eigen::VectorXd mWeightsFullPatchDistance;
		static Eigen::VectorXd mWeightsFullShapeDistance;
		static Eigen::VectorXd mWeightsFullSaliency;

		static Eigen::VectorXd mScaleSimplePatchDistance;
		static Eigen::VectorXd mScaleFullPatchDistance;
		static Eigen::VectorXd mScaleFullShapeDistance;
		static Eigen::VectorXd mScaleFullSaliency;

	private:
		static void error(string s);
	};
}