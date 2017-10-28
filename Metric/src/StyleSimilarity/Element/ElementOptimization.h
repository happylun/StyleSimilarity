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
#include "Data/StyleSimilarityData.h"

using namespace std;

namespace StyleSimilarity {

	class ElementOptimization {

		friend class SymmetryIO;

	public:

		ElementOptimization(StyleSimilarityData *data);
		~ElementOptimization();

	public:

		bool process();
		bool output(string elementFileName);
		bool visualize(string pathName, string affix = "");

	private:

		bool findNearestPoints();
		bool calculateEnergyTerms();
		bool optimizeLabelAssignment();
		bool extractElementParts();

		bool findMatchedPoints(vector<int> &inPatches, int inMode, vector<int> &outPoints);
		bool findMatchedPatches(vector<int> &inPatches, int inMode, vector<int> &outPatches);
		bool verifyEnergyTerms(int modeID);

	protected:

		Eigen::MatrixXd mAssignedUnaryTerm; // unary term value when label is 1 : # of patches X # of modes
		Eigen::MatrixXd mUnassignedUnaryTerm; // unary term value when label is 0 : # of patches X # of modes
		Eigen::MatrixX2i mPairwiseIndex; // patch ID : # of afjacent patch pairs X 2 patches (small ID, large ID)
		Eigen::MatrixXd mPairwiseTerm; // pairwise term value when label is different : # of adjacent patch pairs X # of modes

		vector<int> mElementTransformations; // mode ID : # of elements
		vector<vector<int>> mSourceElementParts; // patch ID : # of patches : # of elements
		vector<vector<int>> mTargetElementParts; // patch ID : # of patches : # of elements

	private:

		static StyleSimilarityData *mpData;

		int mNumSourcePatches;
		int mNumTargetPatches;
		int mNumAdjacentSourcePatches;
		int mNumModes;
		int mNumElements;

		vector<vector<bool>> mOptimizedLabels; // binary labeling result : # of patches : # of modes
		vector<double> mOptimizedUnaryTerms; // unary term after optimization : # of modes
		vector<double> mOptimizedPairwiseTerms; // pairwise term after optimization : # of modes

		Eigen::MatrixXi mSourceNearestPoint; // nearest target point ID : # of source points X # of modes
		Eigen::MatrixXi mTargetNearestPoint; // nearest source point ID : # of target points X # of modes
		vector<int> mSourcePatchMap; // source patch ID : # of source points
		vector<int> mTargetPatchMap; // target patch ID : # of target points

	private:
		static void error(string s);
	};
}