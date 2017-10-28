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

#include "Data/StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class FeatureGeodesic {

		// Geodesic Distances

	public:

		FeatureGeodesic(TSampleSet *samples, vector<double> *features);
		~FeatureGeodesic();
	
	public:

		bool calculate();
		bool visualize(string fileName);

	private:

		bool buildNeighborGraph();
		bool calculateGeodesicDistance();

	private:

		TSampleSet *mpSamples;
		vector<double> *mpFeatures;

		vector<vector<pair<int, float>>> mGraph; // (sample ID, distance) : # of neighbors : # of samples
	};
}