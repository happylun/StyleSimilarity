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

#include "Data/StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class SegmentUtil {

	private:

		// make it non-instantiable
		SegmentUtil() {}
		~SegmentUtil() {}

	public:

		static bool saveSegmentationData(string fileName, vector<vector<int>> &segmentData);
		static bool loadSegmentationData(string fileName, vector<vector<int>> &segmentData);

		static bool savePatchData(string fileName, vector<vector<int>> &patchData, vector<vector<int>> &patchGraph);
		static bool loadPatchData(string fileName, vector<vector<int>> &patchData, vector<vector<int>> &patchGraph);

		static bool visualizeSegmentation(string fileName, TSampleSet &samples, vector<vector<int>> &segments, bool indexedColor = false);
		static bool visualizeSegmentationGraph(string fileName, TSampleSet &samples, vector<vector<int>> &segments, vector<vector<int>> &graph);

		static bool buildKNNGraph(
			TSampleSet &inSamples,
			vector<vector<int>> &outGraph, // sample ID : # of neighbors : # of samples
			vector<bool> &outFlag);        // sample is inliner : # of samples

		static bool extractFinestSegmentation(
			TSampleSet &inSamples,
			vector<vector<int>> &inSegments,
			vector<vector<int>> &outSegments,
			vector<vector<int>> &outSegmentsGraph,
			int level = 0);

		static bool extractPointSet(TPointSet &inSamples, vector<int> &inSegment, TPointSet &outPointSet);
		static bool extractPointSet(TPointSet &inSamples, vector<vector<int>> &inSegments, vector<TPointSet> &outPointSets);
		static bool extractSampleSet(TSampleSet &inSamples, vector<int> &inSegment, TSampleSet &outSampleSet);
		static bool extractSampleSet(TSampleSet &inSamples, vector<vector<int>> &inSegments, vector<TSampleSet> &outSampleSets);

		static bool calculateNormalMap(vector<vec3> &inNormal, vector<double> &outNormalMap);
		static bool compareNormalMap(vector<double> &inSourceNormalMap, vector<double> &inTargetNormalMap, double &outSimilarity);

		static bool labelSampleSet(TSampleSet &inSamples, vector<TTriangleMesh> &inParts, vector<vector<int>> &outSegments);

		static vec3i colorMapping(int index);
	};
}