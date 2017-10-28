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

#include "Library/CMLHelper.h"
#include "Library/SidKDTreeHelper.h"

#include "Data/StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class FeatureCurvature {

		// Curvature

	public:

		struct TCurvature {
			float k1; // max curvature
			float k2; // min curvature
			vec3 d1; // max curvature direction
			vec3 d2; // min curvature direction
			// {n, d1, d2} span an orthonormal right-hand-side coordinate system
		};

	public:

		FeatureCurvature(TSampleSet *samples, vector<TCurvature> *curvatures);
		~FeatureCurvature();
	
	public:

		bool calculate();
		bool visualize(string fileName);
		bool saveFeature(string fileName);

		// 13 metrics
		static bool getDistanceMetrics(TCurvature inCurvature, vector<double> &outMetrics);
		// 7 metrics
		static bool getSaliencyMetrics(TCurvature inCurvature, vector<double> &outMetrics);

		// 4 hist X 13 metrics
		static bool compareFeatures(vector<TCurvature> &curvature1, vector<TCurvature> &curvature2, vector<double> &distance);
		static bool compareFeatures(string file1, string file2, vector<double> &distance);
		static bool loadFeature(string fileName, vector<TCurvature> &curvature);

	private:

		bool buildGeodesicGraph();
		bool getPatchNeighbors();
		bool runPatchFitting();

		inline static double scaleShift(double x) { return (x + 1) / 2; }

	private:

		TSampleSet *mpSamples;
		vector<TCurvature> *mpCurvatures;

		vector<vector<int>> mGraph; // neighbor ID : # of neighbors : # of samples
		vector<vector<int>> mPatch; // point ID : # of points on patch : # of samples
	};
}