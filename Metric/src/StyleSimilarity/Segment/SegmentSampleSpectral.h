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

	class SegmentSampleSpectral {

	public:

		SegmentSampleSpectral(TSampleSet *samples, TTriangleMesh *mesh);
		~SegmentSampleSpectral();

	public:

		bool runSegmentation();
		bool exportSegmentation(vector<vector<int>> &outSegments, vector<vector<int>> &outGraph);
		bool visualizeSegmentation(string fileName);

	private:

		bool visualizeGraph(string fileName, vector<Eigen::Triplet<double, int>> &graph);

	private:

		TSampleSet *mpSamples;
		TTriangleMesh *mpMesh;

		SKDTree mSampleTree;
		SKDTreeData mSampleTreeData;

		vector<vector<int>> mPatches; // point ID : # of sample points : # of patches
		vector<vector<int>> mPatchNeighbors; // neighboring patch ID : # of neighbors : # of patches
	};
}