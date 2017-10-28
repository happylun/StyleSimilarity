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

	class ElementDistance {

	public:

		ElementDistance(StyleSimilarityData *data);
		~ElementDistance();

	public:

		bool loadElement(string elementFileName);

		bool process(string path, string affix);

	private:

		bool computePatchDistance();
		bool computeShapeDistance();

	private:

		StyleSimilarityData *mpData;

		int mNumElements;

		vector<vector<int>> mElementSourcePoints; // point ID : # of points on source element : # of element pairs
		vector<vector<int>> mElementTargetPoints; // point ID : # of points on target element : # of element pairs
		vector<int> mUnmatchSourcePoints; // point ID : # of unmatched points on source shape
		vector<int> mUnmatchTargetPoints; // point ID : # of unmatched points on target shape
		vector<int> mElementTransformations; // mode ID : # of element pairs

		Eigen::MatrixXd mPatchDistance; // # of element pairs X dimPD distance matrix
		Eigen::RowVectorXd mShapeDistance; // dimSD distance vector
		
		//vector<int> mSourceGlobalPatch;
		//vector<int> mTargetGlobalPatch;

	private:
		static void error(string s);
	};
}