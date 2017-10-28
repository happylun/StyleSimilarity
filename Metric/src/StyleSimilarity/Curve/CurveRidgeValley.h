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

#include "Library/CMLHelper.h"
#include "Library/SidKDTreeHelper.h"

#include "Data/StyleSimilarityTypes.h"
#include "Feature/FeatureCurvature.h"

using namespace std;

namespace StyleSimilarity {

	class CurveRidgeValley {

	public:

		CurveRidgeValley();
		~CurveRidgeValley();

		static const int CURVE_TYPES = 4;

	public:

		// SET methods
		inline void loadMesh(TTriangleMesh *mesh) { mpMesh = mesh; }

		bool extractCurve();
		bool visualize(string fileName);
		bool exportPoints(vector<TPointSet> &pointSets, double radius = -1);

	private:

		bool computeCurvature();
		bool computeCurvatureDerivative();
		bool extractZeroCrossings();
		bool extractBoundaryEdges();
		bool chainContours();

		bool visualizeCrossing(string fileName);
		bool visualizeCurvature(string fileName);
		bool visualizeDerivative(string fileName);
		bool visualizeDirection(string fileName);
		bool visualizeBoundary(string fileName);
		bool visualizeChains(string fileName);

	private:

		bool projectCurvatureTensor(const vec3(&oldCS)[3], vec3 oldTensor, const vec3(&newCS)[3], vec3 &newTensor);
		bool projectCurvatureDerivative(const vec3(&oldCS)[3], vec4 oldDerv, const vec3(&newCS)[3], vec4 &newDerv);
		bool computeCornerArea(const vec3(&faceEdge)[3], vec3 &cornerArea);

	private:

		TTriangleMesh *mpMesh;

		vector<FeatureCurvature::TCurvature> mCurvatures;
		vector<vec3> mCurvatureTensors; // 3 unique elements of the 2x2 tensor : # of vertices
		vector<vec4> mCurvatureDerivatives; // 4 unique elements of the 2x2x2 tensor : # of vertices

		TPointSet mCurvePoints;
		vector<bool> mCurvePointTypes; // true-ridge/false-valley : # of curve points
		vector<float> mCurveThickness; // point KMax : # of curve points
		vector<vector<int>> mCurveGraph; // point ID : # of neighbors : # of curve points
		vector<vector<vec2i>> mCurveChains; // curve point ID pair : # of segments : # of chains

		vector<int> mBoundaryEdges; // edge ID
	};
}