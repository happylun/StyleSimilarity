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

#include <vector>
#include <utility>

#include <Eigen/Eigen>

#include "Library/SidKDTreeHelper.h"

#include "StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class FeatureAsset;

	class StyleSimilarityData {

		friend class ElementVoting;
		friend class ElementOptimization;
		friend class ElementDistance;
		friend class ElementMetric;

		friend class DemoIO;

	public:

		StyleSimilarityData();
		~StyleSimilarityData();

	protected:

		TTriangleMesh mSourceMesh;
		TTriangleMesh mTargetMesh;
		TSampleSet mSourceSamples;
		TSampleSet mTargetSamples;

		FeatureAsset *mpSourceFeatures;
		FeatureAsset *mpTargetFeatures;

		vector<TSampleSet> mSourcePatches; // sample set of patch : # of patches
		vector<vector<int>> mSourcePatchesIndices; // sample ID : # of samples in patch : # of patches
		vector<vector<int>> mSourcePatchesGraph; // patch ID : # of neighbors : # of patches

		vector<TSampleSet> mTargetPatches; // sample set of patch : # of patches
		vector<vector<int>> mTargetPatchesIndices; // sample ID : # of samples in patch : # of patches
		vector<vector<int>> mTargetPatchesGraph; // patch ID : # of neighbors : # of patches

		vector<TSampleSet> mSourceSegments; // sample set of segment : # of segments
		vector<vector<int>> mSourceSegmentsIndices; // sample ID : # of samples in segment : # of segments
		vector<TSampleSet> mTargetSegments; // sample set of segment : # of segments
		vector<vector<int>> mTargetSegmentsIndices; // sample ID : # of samples in segment : # of segments

		vector<Eigen::Affine3d> mTransformationModes; // affine transformation matrix : # of transformation modes
		double mDistanceSigma; // sigma for calculating pair weights

		SKDTree mSourceSamplesKdTree;
		SKDTreeData mSourceSamplesKdTreeData;

		SKDTree mTargetSamplesKdTree;
		SKDTreeData mTargetSamplesKdTreeData;
	};

}