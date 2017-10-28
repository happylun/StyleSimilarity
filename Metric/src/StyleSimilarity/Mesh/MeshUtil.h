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

namespace StyleSimilarity {

	class MeshUtil {

	private:

		// make it non-instantiable
		MeshUtil() {}
		~MeshUtil() {}

	public:

		static bool saveMesh(string fileName, TTriangleMesh &mesh);
		static bool loadMesh(string fileName, TTriangleMesh &mesh);

		static bool cleanUp(TTriangleMesh &mesh);

		static bool subdivideMesh(
			TTriangleMesh &inMesh,
			TTriangleMesh &outMesh,
			vector<int> &outFaceIndices,
			double radius = 0);

		static bool subdivideMeshKeepTopology(
			TTriangleMesh &inMesh,
			TTriangleMesh &outMesh,
			vector<int> &outFaceIndices,
			double radius = 0);

		static bool extractMeshFromSamplePoints(
			TTriangleMesh &inMesh,
			TSampleSet &inSamples,
			vector<bool> &inFlags,
			TTriangleMesh &outMesh);

		static bool recomputeNormals(TTriangleMesh &mesh);
		static bool computeDihedralAngle(vec3 center1, vec3 normal1, vec3 center2, vec3 normal2, double &angle);
		static bool computeAABB(TTriangleMesh &mesh, vec3 &bbMin, vec3 &bbMax);
	};
}