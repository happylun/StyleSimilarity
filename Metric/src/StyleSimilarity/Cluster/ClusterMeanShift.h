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

#include "Library/FLANNHelper.h"

using namespace std;

namespace StyleSimilarity {

	class ClusterMeanShift {

	public:

		ClusterMeanShift(vector<vector<double>> &points, vector<double> &weights);
		~ClusterMeanShift();

	public:

		bool setBandwidth(double bandwidth);
		bool runClustering(vector<int> &modes, vector<int> &modeIndices);
		void clearUp();

	private:

		bool iterateMeanShift(int pointID);
		bool doSingleMeanShift(vector<double> &inPoint, vector<double> &outPoint, double &outBandwidth);

		bool perturbCenter(vector<double> &inPoint, vector<double> &outPoint, double bandwidth);
		bool findKNeighbors(vector<double> &inPoint, int k, vector<int> &outIndices, vector<double> &outDistances);
		bool findRangeNeighbors(vector<double> &inPoint, double range, vector<int> &outIndices, vector<double> &outDistances);

	private:

		double kernelProfile(double xNormSq);
		double kernelDerivative(double xNormSq);
		double distance(vector<double> &p1, vector<double> &p2);

	private:

		vector<vector<double>> *mpDataPoints; // high-dimension point : number of points
		vector<double> *mpDataWeights; // weight of point : number of points
		int mNumPoints;
		int mDimensions;
		
		double mFixedBandwidth;
		double mSufficientWeight;
		vector<int> mPointMode; // ID of the mean-shift converging point (as mode) : number of points
		vector<double> mPointBandwidth; // adaptive bandwidth when doing mean-shift on this point : number of points

		// kNN related
		flann::Index<flann::L2<double>> *mpKdTree;
		flann::Matrix<double> *mpKdTreeData;
	};
}