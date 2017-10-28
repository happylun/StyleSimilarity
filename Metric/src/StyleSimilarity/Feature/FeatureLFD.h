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

namespace LFD {
	class LightFieldDescriptor;
}

namespace StyleSimilarity {

	class FeatureLFD {

		// Light Field Descriptor

	public:

		FeatureLFD(TTriangleMesh *mesh, vector<double> *features);
		FeatureLFD(TSampleSet *sample, vector<double> *features);
		~FeatureLFD();
	
	public:

		static bool init();
		static bool finish();

		bool calculate();
		bool visualize(string fileName);

		static bool compareFeatures(vector<double> &feature1, vector<double> &feature2, vector<double> &distances);

	private:

		TTriangleMesh *mpMesh;
		TSampleSet *mpSample;
		vector<double> *mpFeatures;
		LFD::LightFieldDescriptor *mpLFD;

	private:

		static bool mInitialized;
	};
}