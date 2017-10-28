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
#include <string>

#include <Eigen/Sparse>

#include "Library/SidKDTreeHelper.h"

#include "Data/StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class SegmentSampleApxCvx {

	public:

		SegmentSampleApxCvx(TSampleSet *samples, TTriangleMesh *mesh);
		~SegmentSampleApxCvx();

	public:

		bool loadPatches(vector<vector<int>> patches, vector<vector<int>> patchGraph);
		bool runSegmentation();
		bool exportSegmentation(vector<vector<int>> &outSegments);
		bool exportFinestSegmentation(vector<vector<int>> &outSegments, vector<vector<int>> &outGraph);
		bool visualizeSegmentation(string fileName);

	private:

		bool initialize();
		bool computeVisibility();
		bool extractComponents();
		bool mergeSDF();

	private:

		bool subsamplePatches(int patchID, vector<int> &outSamples);
		bool mergeComponents(int compID1, int compID2);
		bool compactComponents();
		bool computeComponentGraph(vector<vector<int>> &outGraph);
		bool exportComponents(vector<vector<int>> &outComponents);

		bool computeSDFDistanceMatrix(vector<vector<int>> &inSegments, Eigen::MatrixXd &outDistanceMatrix, vector<bool> &outFlatFlags);
		bool computeSeamSet(vector<vector<int>> &inSegments, Eigen::MatrixXi &outCVXSeam, Eigen::MatrixXi &outCNCSeam);

	private:

		TSampleSet *mpSamples;
		TTriangleMesh *mpMesh;

		TKDTree mMeshTree;
		TKDTreeData mMeshTreeData;
		SKDTree mSampleTree;
		SKDTreeData mSampleTreeData;

		vector<vector<int>> mPatches; // point ID : # of sample points : # of patches
		vector<vector<int>> mPatchNeighbors; // neighboring patch ID : # of neighbors : # of patches
		vector<vector<double>> mVisibility; // visibility in [0,1] : (# of patches) * (# of patches)

		vector<int> mComponentIndices; // component ID : # of patches
		vector<vector<int>> mComponentPatches; // patch ID : # of patches in component : # of components

		vector<vector<int>> mSegments; // point ID : # of sample points : # of segments (mixed results)
		vector<int> mSegmentsOffset; // offset in mSegments : # of different segmentation results

		vector<vector<int>> mFinestSegment; // point ID : # of sample points : # of segments (finest resolution segmentation)
		vector<vector<int>> mFinestSegmentGraph; // neighbor segment ID : # of neighbors : # of segments

		vector<vector<int>> mSampleGraph; // neighbor point ID : # of neighbors : # of points
		vector<bool> mSampleFlag; // valid in kNN graph : # of points
	};
}