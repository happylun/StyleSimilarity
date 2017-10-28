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

#include "Library/FLANNHelper.h"

using namespace std;

namespace StyleSimilarity {

	// No offense. I am just using this name because
	// it's compact and easy to pronounce/remember.
	class FeatureTalSaliency {

		// Saliency Detection in Large Point Sets [13-SLT]

	public:

		// [saliencies][OUT] saliency metric: # metrics : # sample points
		FeatureTalSaliency(TSampleSet *samples, vector<vector<double>> *saliencies);
		~FeatureTalSaliency();

		template<class FeatureClass> bool calculate() {
			if (!FeatureClass::calculate(*mpSamples, *mpSamples, mLowFeatures, mLowRadius)) return false;
			if (!calculateLowLevelSaliencies()) return false;
			if (!FeatureClass::calculate(*mpSamples, *mpSamples, mHighFeatures, mHighRadius)) return false;
			if (!calculateOtherSaliencies()) return false;
			if (!exportSaliencies()) return false;
			return true;
		}

		bool visualize(string fileName);
		bool saveFeature(string fileName);

		static bool loadFeature(string fileName, vector<vector<double>> &feature);
	
	private:

		bool calculateLowLevelSaliencies();
		bool calculateOtherSaliencies();
		bool exportSaliencies();

		bool extractDistinctPoints();
		bool calculateDLow();
		bool calculateALow();
		bool calculateDHigh();

	private:

		TSampleSet *mpSamples; // input
		vector<vector<double>> *mpSaliencies; // output

		int mNumPoints;
		int mNumBins;

		TSampleSet mDistinctPoints;

		vector<vector<double>> mLowFeatures;
		vector<vector<double>> mHighFeatures;

		vector<double> mSaliencyDLow;
		vector<double> mSaliencyALow;
		vector<double> mSaliencyDHigh;
		
		vec3 mBBMin;
		vec3 mBBMax;
		double mLowRadius;
		double mHighRadius;
	};
}
