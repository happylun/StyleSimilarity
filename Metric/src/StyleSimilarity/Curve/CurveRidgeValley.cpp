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

#include "CurveRidgeValley.h"

#include <iostream>
#include <fstream>
#include <map>
#include <set>

#include "Eigen/Eigen"

#include "Utility/PlyExporter.h"
#include "Utility/PlyLoader.h"

#include "Data/StyleSimilarityConfig.h"

using namespace std;
using namespace StyleSimilarity;

CurveRidgeValley::CurveRidgeValley() {

	mpMesh = 0;
}

CurveRidgeValley::~CurveRidgeValley() {

}

bool CurveRidgeValley::extractCurve() {

	if (!mpMesh) {
		cout << "Error: data not fully loaded" << endl;
		return false;
	}

	cout << "computing curvature" << endl;
	if (!computeCurvature()) return false;
	cout << "computing curvature derivative" << endl;
	if (!computeCurvatureDerivative()) return false;
	cout << "extracting zero crossings" << endl;
	if (!extractZeroCrossings()) return false;
	cout << "extracting boundaries" << endl;
	if (!extractBoundaryEdges()) return false;
	cout << "chaining curves" << endl;
	if (!chainContours()) return false;
	cout << "done" << endl;

	return true;
}

bool CurveRidgeValley::computeCurvature() {

	int numVertices = mpMesh->amount;
	int numFaces = (int)mpMesh->indices.size();

	vector<float> totalWeights(numVertices, 0.0f);
	mCurvatureTensors.clear();
	mCurvatureTensors.resize(numVertices, vec3(0.0f, 0.0f, 0.0f));

	vector<vec3> vertexDir1(numVertices), vertexDir2(numVertices);
	vector<bool> vertexFlag(numVertices, false);
#pragma omp parallel for
	for (int faceID = 0; faceID < numFaces; faceID++) {
		vec3i faceIdx = mpMesh->indices[faceID];
		for (int j = 0; j < 3; j++) {
			vertexDir1[faceIdx[j]] = mpMesh->positions[faceIdx[(j + 1) % 3]]
				- mpMesh->positions[faceIdx[j]];
			vertexFlag[faceIdx[j]] = true;
		}
	}
#pragma omp parallel for
	for (int vertID = 0; vertID < numVertices; vertID++) {
		if (vertexFlag[vertID]) {
			vec3 vertexN = mpMesh->normals[vertID];
			vertexDir1[vertID] = cml::normalize(cml::cross(vertexDir1[vertID], vertexN));
			vertexDir2[vertID] = cml::normalize(cml::cross(vertexN, vertexDir1[vertID]));
		}
	}

#pragma omp parallel for
	for (int faceID = 0; faceID < numFaces; faceID++) {

		vec3i faceIdx = mpMesh->indices[faceID];

		vec3 vertexP[3], vertexN[3];
		for (int j = 0; j < 3; j++) {
			vertexP[j] = mpMesh->positions[faceIdx[j]];
			vertexN[j] = cml::normalize(mpMesh->normals[faceIdx[j]]);
		}
		vec3 faceEdge[] = { vertexP[2] - vertexP[1], vertexP[0] - vertexP[2], vertexP[1] - vertexP[0] };

		vec3 cornerArea;
		if (!computeCornerArea(faceEdge, cornerArea)) continue; // degenerated face

		// N-T-B coordinate system
		vec3 faceCS[3];
		faceCS[1] = cml::normalize(faceEdge[0]);
		faceCS[0] = cml::normalize(cml::cross(faceCS[1], cml::normalize(faceEdge[1])));
		faceCS[2] = cml::normalize(cml::cross(faceCS[0], faceCS[1]));

		// build linear system
		Eigen::MatrixXf matA = Eigen::MatrixXf::Zero(6, 3);
		Eigen::VectorXf matB = Eigen::MatrixXf::Zero(6, 1);
		for (int j = 0; j < 3; j++) {
			float u = cml::dot(faceEdge[j], faceCS[1]);
			float v = cml::dot(faceEdge[j], faceCS[2]);
			float nu = cml::dot(vertexN[(j + 2) % 3] - vertexN[(j + 1) % 3], faceCS[1]);
			float nv = cml::dot(vertexN[(j + 2) % 3] - vertexN[(j + 1) % 3], faceCS[2]);

			matA.row(j * 2 + 0) << u, v, 0;
			matA.row(j * 2 + 1) << 0, u, v;

			matB.row(j * 2 + 0) << nu;
			matB.row(j * 2 + 1) << nv;
		}

		// solve linear system: A * X = B
		if (!matA.allFinite() || !matB.allFinite()) {
			cout << "Error: matrix is invalid!" << endl;
		}
		Eigen::VectorXf matX = matA.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(matB);
		vec3 faceTensor = vec3(matX(0), matX(1), matX(2));

		// project back to vertex CS
		vec3 vertexTensor[3];
		for (int j = 0; j < 3; j++) {
			vec3 vertexCS[] = {vertexN[j], vertexDir1[faceIdx[j]], vertexDir2[faceIdx[j]]};
			if (!projectCurvatureTensor(faceCS, faceTensor, vertexCS, vertexTensor[j])) {
				cout << "Error: projectCurvatureTensor" << endl;
			}
		}

		// add weighted tensor
#pragma omp critical
		{
			for (int j = 0; j < 3; j++) {
				mCurvatureTensors[faceIdx[j]] += vertexTensor[j] * cornerArea[j];
				totalWeights[faceIdx[j]] += cornerArea[j];
			}
		}
	}

	mCurvatures.clear();
	mCurvatures.resize(numVertices);

	// extract principal curvature and direction from tensor
#pragma omp parallel for
	for (int vertexID = 0; vertexID < numVertices; vertexID++) {

		if (totalWeights[vertexID] > 0) {
			mCurvatureTensors[vertexID] /= totalWeights[vertexID]; // normalize

			// curvature tensor
			vec3 tensor = mCurvatureTensors[vertexID];
			Eigen::Matrix2f matK;
			matK << tensor[0], tensor[1], tensor[1], tensor[2];

			// solve for eigen values and eigen vectors
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigenSolver(matK);
			if (eigenSolver.info() != Eigen::Success) {
				cout << "Error: fail to solve for principal curvatures" << endl;
			}

			vec3 localX = vertexDir1[vertexID];
			vec3 localY = vertexDir2[vertexID];
			vec3 localZ = cml::normalize(mpMesh->normals[vertexID]);

			FeatureCurvature::TCurvature &curvature = mCurvatures[vertexID];
			curvature.k1 = eigenSolver.eigenvalues()(1);
			curvature.k2 = eigenSolver.eigenvalues()(0);
			curvature.d1 = cml::normalize(localX * eigenSolver.eigenvectors()(0, 1) + localY * eigenSolver.eigenvectors()(1, 1));
			curvature.d2 = cml::normalize(localX * eigenSolver.eigenvectors()(0, 0) + localY * eigenSolver.eigenvectors()(1, 0));
			
			if (fabs(curvature.k1) < fabs(curvature.k2)) {
				swap(curvature.k1, curvature.k2);
				swap(curvature.d1, curvature.d2);
			}

			// re-orient curvature direction to make {n, d1, d2} a right-hand-side coordinate system
			if (cml::dot(cml::cross(curvature.d1, curvature.d2), localZ) < 0) curvature.d2 = -curvature.d2;
		} else {
			cout << "Error: zero weight for curvature tensor" << endl;
		}
	}

	return true;
}

bool CurveRidgeValley::computeCurvatureDerivative() {

	int numVertices = mpMesh->amount;
	int numFaces = (int)mpMesh->indices.size();

	vector<float> totalWeights(numVertices, 0.0f);
	mCurvatureDerivatives.clear();
	mCurvatureDerivatives.resize(numVertices, vec4(0.0f, 0.0f, 0.0f, 0.0f));

#pragma omp parallel for
	for (int faceID = 0; faceID < numFaces; faceID++) {

		vec3i faceIdx = mpMesh->indices[faceID];

		vec3 vertexP[3], vertexN[3];
		for (int j = 0; j < 3; j++) {
			vertexP[j] = mpMesh->positions[faceIdx[j]];
			vertexN[j] = cml::normalize(mpMesh->normals[faceIdx[j]]);
		}
		vec3 faceEdge[] = { vertexP[2] - vertexP[1], vertexP[0] - vertexP[2], vertexP[1] - vertexP[0] };

		vec3 cornerArea;
		if (!computeCornerArea(faceEdge, cornerArea)) continue; // degenerated face

		// curvature tensor
		FeatureCurvature::TCurvature vertexCurvatures[3];
		for (int j = 0; j < 3; j++) {
			vertexCurvatures[j] = mCurvatures[faceIdx[j]];
		}

		// N-T-B coordinate system
		vec3 faceCS[3];
		faceCS[1] = cml::normalize(faceEdge[0]);
		faceCS[0] = cml::normalize(cml::cross(faceCS[1], cml::normalize(faceEdge[1])));
		faceCS[2] = cml::normalize(cml::cross(faceCS[0], faceCS[1]));

		// project tensor to face's CS
		vec3 faceTensor[3];
		for (int j = 0; j < 3; j++) {
			vec3 vertexCS[] = { vertexN[j], vertexCurvatures[j].d1, vertexCurvatures[j].d2 };
			vec3 vertexTensor = vec3(vertexCurvatures[j].k1, 0.0f, vertexCurvatures[j].k2);
			if (!projectCurvatureTensor(vertexCS, vertexTensor, faceCS, faceTensor[j])) {
				cout << "Error: projectCurvatureTensor" << endl;
			}
		}

		// build linear system
		Eigen::MatrixXf matA = Eigen::MatrixXf::Zero(12, 4);
		Eigen::VectorXf matB = Eigen::MatrixXf::Zero(12, 1);
		for (int j = 0; j < 3; j++) {
			float u = cml::dot(faceEdge[j], faceCS[1]);
			float v = cml::dot(faceEdge[j], faceCS[2]);
			vec3 d = faceTensor[(j + 2) % 3] - faceTensor[(j + 1) % 3];

			matA.row(j * 4 + 0) << u, v, 0, 0;
			matA.row(j * 4 + 1) << 0, u, v, 0;
			matA.row(j * 4 + 2) << 0, u, v, 0;
			matA.row(j * 4 + 3) << 0, 0, u, v;

			matB.row(j * 4 + 0) << d[0];
			matB.row(j * 4 + 1) << d[1];
			matB.row(j * 4 + 2) << d[1];
			matB.row(j * 4 + 3) << d[2];
		}

		// solve linear system: A * X = B
		if (!matA.allFinite() || !matB.allFinite()) {
			//cout << "Error: matrix is invalid!" << endl;
		}
		Eigen::VectorXf matX = matA.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(matB);
		vec4 faceDerv = vec4(matX(0), matX(1), matX(2), matX(3));

		// project back to vertex CS
		vec4 vertexDerv[3];
		for (int j = 0; j < 3; j++) {
			vec3 vertexCS[] = { vertexN[j], vertexCurvatures[j].d1, vertexCurvatures[j].d2 };
			if (!projectCurvatureDerivative(faceCS, faceDerv, vertexCS, vertexDerv[j])) {
				cout << "Error: projectCurvatureDerivative" << endl;
			}
		}

		// add weighted derivation
#pragma omp critical
		{
			for (int j = 0; j < 3; j++) {
				mCurvatureDerivatives[faceIdx[j]] += vertexDerv[j] * cornerArea[j];
				totalWeights[faceIdx[j]] += cornerArea[j];
			}
		}
	}

	// normalize
#pragma omp parallel for
	for (int vertexID = 0; vertexID < numVertices; vertexID++) {
		if (totalWeights[vertexID] > 0) {
			mCurvatureDerivatives[vertexID] /= totalWeights[vertexID];
		}
	}

	return true;
}

bool CurveRidgeValley::extractZeroCrossings() {

	const float thicknessThreshold = 0.0f; // UNDONE: param KMax of zero crossing point

	mCurvePoints.positions.clear();
	mCurvePoints.normals.clear();
	mCurvePoints.amount = 0;

	mCurvePointTypes.clear();
	mCurveThickness.clear();
	mCurveGraph.clear();

	map<vec2i, int> edgeMap;

	for (vec3i faceIdx : mpMesh->indices) { // check all faces

		// curvature related properties
		vec3 vertexP[3], vertexN[3];
		float vertexKMax[3], vertexKMin[3]; // curvature
		float vertexEMax[3], vertexEMin[3]; // curvature derivative along curvature direction
		vec3 vertexTMax[3], vertexTMin[3]; // curvature direction towards increasing curvature
		for (int j = 0; j < 3; j++) {
			int vertexID = faceIdx[j];
			vertexP[j] = mpMesh->positions[vertexID];
			vertexN[j] = cml::normalize(mpMesh->normals[vertexID]);
			vertexKMax[j] = mCurvatures[vertexID].k1;
			vertexKMin[j] = mCurvatures[vertexID].k2;
			vertexEMax[j] = mCurvatureDerivatives[vertexID][0];
			vertexEMin[j] = mCurvatureDerivatives[vertexID][3];
			vertexTMax[j] = mCurvatures[vertexID].d1 * vertexEMax[j];
			vertexTMin[j] = mCurvatures[vertexID].d2 * vertexEMin[j];
		}

		// find zero crossing points on edges
		vec3i pointIdx;
		bool pointTypes[3];
		for (int j = 0; j < 3; j++) { // check all 3 edges

			int i1 = j, i2 = (j + 1) % 3;
			int vID1 = faceIdx[i1];
			int vID2 = faceIdx[i2];
			if (vID1>vID2) swap(vID1, vID2);
			vec2i vKey = vec2i(vID1, vID2);
			auto it = edgeMap.find(vKey);
			if (it == edgeMap.end()) { // not checked yet

				bool type = false;
				float alpha = -1; // it won't find ridge point and valley at the same time

				// find ridge point
				if (vertexKMax[i1] > fabs(vertexKMin[i1]) && // ridge
					vertexKMax[i2] > fabs(vertexKMin[i2]) && // ridge
					cml::dot(vertexTMax[i1], vertexTMax[i2]) <= 0 && // zero crosing
					(cml::dot(vertexP[i2] - vertexP[i1], vertexTMax[i1]) >= 0 || // maximum
					cml::dot(vertexP[i1] - vertexP[i2], vertexTMax[i2]) >= 0) // maximum
					)
				{
					type = true;
					alpha = fabs(vertexEMax[i2]) / (fabs(vertexEMax[i1]) + fabs(vertexEMax[i2]));					
				}

				// find valley point
				if (vertexKMax[i1] < -fabs(vertexKMin[i1]) && // valley
					vertexKMax[i2] < -fabs(vertexKMin[i2]) && // valley
					cml::dot(vertexTMax[i1], vertexTMax[i2]) <= 0 && // zero crosing
					(cml::dot(vertexP[i2] - vertexP[i1], vertexTMax[i1]) <= 0 || // minimum
					cml::dot(vertexP[i1] - vertexP[i2], vertexTMax[i2]) <= 0) // minimum
					)
				{
					type = false;
					alpha = fabs(vertexEMax[i2]) / (fabs(vertexEMax[i1]) + fabs(vertexEMax[i2]));
				}

				// interpolate
				if (alpha >= 0.0f && alpha <= 1.0f) {					
					vec3 zeroP = vertexP[i1] * alpha + vertexP[i2] * (1 - alpha);
					vec3 zeroN = cml::normalize(vertexN[i1] * alpha + vertexN[i2] * (1 - alpha));
					float zeroKMax = vertexKMax[i1] * alpha + vertexKMax[i2] * (1 - alpha);

					int pointID = (int)mCurvePoints.positions.size();
					mCurvePoints.positions.push_back(zeroP);
					mCurvePoints.normals.push_back(zeroN);
					mCurvePointTypes.push_back(type);
					mCurveThickness.push_back(fabs(zeroKMax));
					mCurveGraph.push_back(vector<int>());
					edgeMap[vKey] = pointID;
					pointIdx[j] = pointID;
					pointTypes[j] = type;
				} else {
					pointIdx[j] = -1;
					pointTypes[j] = false;
				}
			} else { // already checked
				pointIdx[j] = it->second;
				pointTypes[j] = mCurvePointTypes[it->second];
			}
		}

		// connect points
		if (pointIdx[0] >= 0 && pointIdx[1] >= 0 && pointIdx[2] >= 0 &&
			pointTypes[0] == pointTypes[1] && pointTypes[0] == pointTypes[2])
		{
			// connect all three curve points to center point
			vec3 centerP = (vertexP[0] + vertexP[1] + vertexP[2]) / 3;
			vec3 centerN = cml::normalize(vertexN[0] + vertexN[1] + vertexN[2]);
			float centerThickness = (mCurveThickness[pointIdx[0]] + mCurveThickness[pointIdx[1]] + mCurveThickness[pointIdx[2]]) / 3;
			int centerID = (int)mCurvePoints.positions.size();
			bool centerType = ((int)pointTypes[0] + (int)pointTypes[1] + (int)pointTypes[2]) > 1;
			mCurvePoints.positions.push_back(centerP);
			mCurvePoints.normals.push_back(centerN);
			mCurvePointTypes.push_back(centerType);
			mCurveThickness.push_back(centerThickness);
			mCurveGraph.push_back(vector<int>());
			for (int j = 0; j < 3; j++) {
				if ((mCurveThickness[pointIdx[j]] + centerThickness) / 2 >= thicknessThreshold) {
					mCurveGraph[pointIdx[j]].push_back(centerID);
					mCurveGraph[centerID].push_back(pointIdx[j]);
				}
			}
		} else {
			// connect two curve points
			for (int j = 0; j < 3; j++) {
				int i1 = j, i2 = (j + 1) % 3;
				if (pointIdx[i1] >= 0 && pointIdx[i2] >= 0 && pointTypes[i1] == pointTypes[i2]) {
					if ((mCurveThickness[pointIdx[i1]] + mCurveThickness[pointIdx[i2]]) / 2 >= thicknessThreshold) {
						mCurveGraph[pointIdx[i1]].push_back(pointIdx[i2]);
						mCurveGraph[pointIdx[i2]].push_back(pointIdx[i1]);
					}
				}
			}
		}
	}

	if (mCurvePoints.positions.size() != mCurvePoints.normals.size() ||
		mCurvePoints.positions.size() != mCurvePointTypes.size())
	{
		cout << "Error: size of curve points incorrect" << endl;
		return false;
	}
	mCurvePoints.amount = (int)mCurvePoints.positions.size();

	return true;
}

bool CurveRidgeValley::extractBoundaryEdges() {

	int numFaces = (int)mpMesh->indices.size();

	map<vec2i, int> bMap; // (vertex index 1, vertex index 2) : edge ID (boundary) or -1 (not boundary)
	set<vec3i> fSet; // face set (increasing vertex order - ignore original vertex order)
	for(int faceID=0; faceID<numFaces; faceID++) {
		vec3i faceIdx = mpMesh->indices[faceID];
		vec3i faceKey = faceIdx;
		if (faceKey[0] > faceKey[1]) swap(faceKey[0], faceKey[1]);
		if (faceKey[0] > faceKey[2]) swap(faceKey[0], faceKey[2]);
		if (faceKey[1] > faceKey[2]) swap(faceKey[1], faceKey[2]);
		if (fSet.find(faceKey) == fSet.end()) {
			fSet.insert(faceKey);
		} else {
			continue;
		}
		for(int j=0; j<3; j++) {
			vec2i key = vec2i(faceIdx[j], faceIdx[(j+1)%3]);
			if(key[0] > key[1]) swap(key[0], key[1]);
			auto it = bMap.find(key);
			if(it == bMap.end()) {
				bMap[key] = faceID*3+j;
			} else {
				it->second = -1;
			}
		}
	}

	set<int> bSet; // edge ID
	for(auto it : bMap) {
		if(it.second >= 0) {
			bSet.insert(it.second);
		}
	}

	mBoundaryEdges.clear();
	mBoundaryEdges.insert(mBoundaryEdges.end(), bSet.begin(), bSet.end());

	return true;
}

bool CurveRidgeValley::chainContours() {

	const float strengthThreshold = (float)StyleSimilarityConfig::mCurve_RVStrengthThreshold;
	const float lengthThreshold = (float)StyleSimilarityConfig::mCurve_RVLengthThreshold;

	mCurveChains.clear();

	int numPoints = mCurvePoints.amount;

	vector<bool> pointFlag(numPoints, false);
	vector<int> pointQueue;

	// find connected components
	for (int pointID = 0; pointID < numPoints; pointID++) {

		if (pointFlag[pointID]) continue;
		pointFlag[pointID] = true;
		pointQueue.clear();
		pointQueue.push_back(pointID);

		vector<vec2i> chain;
		float chainStrenth = 0;
		float chainLength = 0;

		int head = 0;
		while (head < (int)pointQueue.size()) { // bfs
			for (int neighborID : mCurveGraph[pointQueue[head]]) {

				if (pointFlag[neighborID]) continue;
				pointFlag[neighborID] = true;
				pointQueue.push_back(neighborID);

				int i1 = pointQueue[head];
				int i2 = neighborID;
				vec3 p1 = mCurvePoints.positions[i1];
				vec3 p2 = mCurvePoints.positions[i2];
				float k1 = mCurveThickness[i1];
				float k2 = mCurveThickness[i2];
				float length = (p1 - p2).length();
				float strenth = (k1 + k2) / 2 * length;

				chain.push_back(vec2i(i1, i2));
				chainStrenth += strenth;
				chainLength += length;
			}
			head++;
		}

		if (chainStrenth >= strengthThreshold && chainLength >= lengthThreshold) {
			mCurveChains.push_back(chain);
		}
	}

	return true;
}

bool CurveRidgeValley::projectCurvatureTensor(
	const vec3(&oldCS)[3], vec3 oldTensor,
	const vec3(&newCS)[3], vec3 &newTensor)
{
	// ref: TriMesh_curvature.cc

	// rotate new CS to be perp to normal of old CS
	float normalCos = cml::dot(oldCS[0], newCS[0]);
	vec3 newU, newV;
	if (normalCos > -1.0f) {
		vec3 perp = oldCS[0] - newCS[0] * normalCos;
		vec3 dperp = (oldCS[0] + newCS[0]) / (normalCos + 1);
		newU = newCS[1] - dperp * cml::dot(newCS[1], perp);
		newV = newCS[2] - dperp * cml::dot(newCS[2], perp);
	} else {
		newU = -newCS[1];
		newV = -newCS[2];
	}

	// reproject curvature tensor
	float u1 = cml::dot(newU, oldCS[1]);
	float v1 = cml::dot(newU, oldCS[2]);
	float u2 = cml::dot(newV, oldCS[1]);
	float v2 = cml::dot(newV, oldCS[2]);
	newTensor[0] = oldTensor[0] * u1*u1 + oldTensor[1] * ( 2.0f*u1*v1) + oldTensor[2] * v1*v1;
	newTensor[1] = oldTensor[0] * u1*u2 + oldTensor[1] * (u1*v2+u2*v1) + oldTensor[2] * v1*v2;
	newTensor[2] = oldTensor[0] * u2*u2 + oldTensor[1] * ( 2.0f*u2*v2) + oldTensor[2] * v2*v2;

	return true;
}

bool CurveRidgeValley::projectCurvatureDerivative(
	const vec3(&oldCS)[3], vec4 oldDerv,
	const vec3(&newCS)[3], vec4 &newDerv)
{
	// ref: TriMesh_curvature.cc

	// rotate new CS to be perp to normal of old CS
	float normalCos = cml::dot(oldCS[0], newCS[0]);
	vec3 newU, newV;
	if (normalCos > -1.0f) {
		vec3 perp = oldCS[0] - newCS[0] * normalCos;
		vec3 dperp = (oldCS[0] + newCS[0]) / (normalCos + 1);
		newU = newCS[1] - dperp * cml::dot(newCS[1], perp);
		newV = newCS[2] - dperp * cml::dot(newCS[2], perp);
	} else {
		newU = -newCS[1];
		newV = -newCS[2];
	}

	// reproject derivative tensor
	float u1 = cml::dot(newU, oldCS[1]);
	float v1 = cml::dot(newU, oldCS[2]);
	float u2 = cml::dot(newV, oldCS[1]);
	float v2 = cml::dot(newV, oldCS[2]);
	newDerv[0] = oldDerv[0] * u1*u1*u1
		+ oldDerv[1] * 3.0f*u1*u1*v1
		+ oldDerv[2] * 3.0f*u1*v1*v1
		+ oldDerv[3] * v1*v1*v1;
	newDerv[1] = oldDerv[0] * u1*u1*u2
		+ oldDerv[1] * (u1*u1*v2+2.0f*u2*u1*v1)
		+ oldDerv[2] * (u2*v1*v1 + 2.0f*u1*v1*v2)
		+ oldDerv[3] * v1*v1*v2;
	newDerv[2] = oldDerv[0] * u1*u2*u2
		+ oldDerv[1] * (u2*u2*v1 + 2.0f*u1*u2*v2)
		+ oldDerv[2] * (u1*v2*v2 + 2.0f*u2*v2*v1)
		+ oldDerv[3] * v1*v2*v2;
	newDerv[3] = oldDerv[0] * u2*u2*u2
		+ oldDerv[1] * 3.0f*u2*u2*v2
		+ oldDerv[2] * 3.0f*u2*v2*v2
		+ oldDerv[3] * v2*v2*v2;

	return true;
}

bool CurveRidgeValley::computeCornerArea(const vec3(&faceEdge)[3], vec3 &cornerArea) {

	// ref: TriMesh_pointareas.cc

	float faceArea = cml::cross(faceEdge[0], faceEdge[1]).length() / 2;
	if (faceArea <= 0) return false; // degenerated face

	float lengthSq[3];
	for (int k = 0; k < 3; k++) {
		lengthSq[k] = faceEdge[k].length_squared();
	}
	float edgeWeight[3];
	for (int k = 0; k < 3; k++) {
		edgeWeight[k] = lengthSq[k] * (lengthSq[(k + 1) % 3] + lengthSq[(k + 2) % 3] - lengthSq[k]);
		if (edgeWeight[k] <= 0) {
			cornerArea[(k + 1) % 3] = -0.25f * lengthSq[(k + 2) % 3] * faceArea / cml::dot(faceEdge[k], faceEdge[(k + 2) % 3]);
			cornerArea[(k + 2) % 3] = -0.25f * lengthSq[(k + 1) % 3] * faceArea / cml::dot(faceEdge[k], faceEdge[(k + 1) % 3]);
			cornerArea[k] = faceArea - cornerArea[(k + 1) % 3] - cornerArea[(k + 2) % 3];
			return true;
		}
	}
	float edgeWeightScale = 0.5f * faceArea / (edgeWeight[0] + edgeWeight[1] + edgeWeight[2]);
	for (int k = 0; k < 3; k++) {
		cornerArea[k] = edgeWeightScale * (edgeWeight[(k + 1) % 3] + edgeWeight[(k + 2) % 3]);
	}

	return true;
}

bool CurveRidgeValley::exportPoints(vector<TPointSet> &pointSets, double radius) {

	if (radius < 0) {
		radius = StyleSimilarityConfig::mCurve_RVPointSamplingRadius;
	}

	TPointSet allLines[CURVE_TYPES];
	TPointSet &ridgeLines = allLines[0];
	TPointSet &valleyLines = allLines[1];
	TPointSet &boundaryLines = allLines[2];
	TPointSet &mixLines = allLines[3];

	// accumulate line segments on ridges & valleys
	for (auto & chain : mCurveChains) {
		for (vec2i segID : chain) {
			vec3 p1 = mCurvePoints.positions[segID[0]];
			vec3 p2 = mCurvePoints.positions[segID[1]];
			vec3 n1 = mCurvePoints.normals[segID[0]];
			vec3 n2 = mCurvePoints.normals[segID[1]];
			bool t1 = mCurvePointTypes[segID[0]];
			bool t2 = mCurvePointTypes[segID[1]];
			mixLines.positions.push_back(p1);
			mixLines.positions.push_back(p2);
			mixLines.normals.push_back(n1);
			mixLines.normals.push_back(n2);
			if (t1 || t2) {
				ridgeLines.positions.push_back(p1);
				ridgeLines.positions.push_back(p2);
				ridgeLines.normals.push_back(n1);
				ridgeLines.normals.push_back(n2);
			}
			if (!t1 || !t2) {
				valleyLines.positions.push_back(p1);
				valleyLines.positions.push_back(p2);
				valleyLines.normals.push_back(n1);
				valleyLines.normals.push_back(n2);
			}
		}
	}

	// accumulate line segments on boundaries
	for (int edgeID : mBoundaryEdges) {
		vec3i faceIdx = mpMesh->indices[edgeID / 3];
		int vID1 = faceIdx[edgeID % 3];
		int vID2 = faceIdx[(edgeID + 1) % 3];
		vec3 p1 = mpMesh->positions[vID1];
		vec3 p2 = mpMesh->positions[vID2];
		vec3 n1 = mpMesh->normals[vID1];
		vec3 n2 = mpMesh->normals[vID2];

		mixLines.positions.push_back(p1);
		mixLines.positions.push_back(p2);
		mixLines.normals.push_back(n1);
		mixLines.normals.push_back(n2);

		boundaryLines.positions.push_back(p1);
		boundaryLines.positions.push_back(p2);
		boundaryLines.normals.push_back(n1);
		boundaryLines.normals.push_back(n2);
	}

	// export point sets for each type

	pointSets.resize(CURVE_TYPES);
	for (int lineType = 0; lineType < CURVE_TYPES; lineType++) {
		TPointSet &pointSet = pointSets[lineType];

		pointSet.positions.clear();
		pointSet.normals.clear();
		int numSegments = (int)allLines[lineType].positions.size() / 2;
		for (int segmentID = 0; segmentID < numSegments; segmentID++) {
			vec3 v1 = allLines[lineType].positions[segmentID * 2];
			vec3 v2 = allLines[lineType].positions[segmentID * 2 + 1];
			vec3 n1 = allLines[lineType].normals[segmentID * 2];
			vec3 n2 = allLines[lineType].normals[segmentID * 2 + 1];

			// equal-distance point sampling on line segments
			float l = (v1 - v2).length();
			int m = (int)(l / radius);
			for (int j = 0; j < m; j++) {
				float a = (float)(radius*j / l);
				vec3 vp = v2 * a + v1 * (1 - a);
				vec3 vn = cml::normalize(n2 * a + n1 * (1 - a));
				pointSet.positions.push_back(vp);
				pointSet.normals.push_back(vn);
			}
		}
		pointSet.amount = (int)pointSet.positions.size();
	}

	return true;
}

bool CurveRidgeValley::visualize(string fileName) {

	string boundaryName = fileName.substr(0, fileName.find_last_of('.')) + "-boundary.ply";
	if (!visualizeBoundary(boundaryName)) return false;

	string crossingName = fileName.substr(0, fileName.find_last_of('.')) + "-crossing.ply";
	if (!visualizeCrossing(crossingName)) return false;

	string chainsName = fileName.substr(0, fileName.find_last_of('.')) + "-chains.ply";
	if (!visualizeChains(chainsName)) return false;

	string curvName = fileName.substr(0, fileName.find_last_of('.')) + "-curvature.ply";
	if (!visualizeCurvature(curvName)) return false;

	string dervName = fileName.substr(0, fileName.find_last_of('.')) + "-derivative.ply";
	if (!visualizeDerivative(dervName)) return false;

	string dirName = fileName.substr(0, fileName.find_last_of('.')) + "-direction.ply";
	if (!visualizeDirection(dirName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeCurvature(string fileName) {

	PlyExporter pe;

	// export curvature direction as thin rects

	vector<vec3i> vI; // indices of a rect
	vI.push_back(vec3i(0, 2, 1));
	vI.push_back(vec3i(1, 2, 3));

	int numPoints = mpMesh->amount;
	float radius = 0.005f; // HACK: manual radius

	vector<float> allAbsK1;
	vector<float> allAbsK2;
	for (FeatureCurvature::TCurvature &curvature : mCurvatures) {
		allAbsK1.push_back(fabs(curvature.k1));
		allAbsK2.push_back(fabs(curvature.k2));
	}
	int n = (int)(numPoints * 0.8);
	nth_element(allAbsK1.begin(), allAbsK1.begin() + n, allAbsK1.end());
	nth_element(allAbsK2.begin(), allAbsK2.begin() + n, allAbsK2.end());
	float maxAbsK1 = allAbsK1[n];
	float maxAbsK2 = allAbsK2[n];

	for (int pointID = 0; pointID < numPoints; pointID++) {

		vec3 position = mpMesh->positions[pointID];
		FeatureCurvature::TCurvature &curvature = mCurvatures[pointID];

		vec3 p0 = position + (curvature.d2.normalize() + curvature.d1.normalize() * 0.2f) * radius * 2.0f;
		vec3 p1 = position + (-curvature.d2.normalize() + curvature.d1.normalize() * 0.2f) * radius * 2.0f;
		vec3 p2 = position + (curvature.d2.normalize() - curvature.d1.normalize() * 0.2f) * radius * 2.0f;
		vec3 p3 = position + (-curvature.d2.normalize() - curvature.d1.normalize() * 0.2f) * radius * 2.0f;

		vector<vec3> vP;
		vP.push_back(p0);
		vP.push_back(p1);
		vP.push_back(p2);
		vP.push_back(p3);

		float c = cml::clamp(curvature.k2 / maxAbsK2, -1.0f, 1.0f);
		int c1 = max(0, (int)(c * 255));
		int c2 = max(0, (int)(-c * 255));
		vec3i color = vec3i(255 - c2, 255 - c1 - c2, 255 - c1);

		if (!pe.addMesh(&vI, &vP, 0, cml::identity_4x4(), color)) return false;
	}

	if (!pe.output(fileName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeDerivative(string fileName) {

	int numPoints = mpMesh->amount;

	vector<vec3i> meshColor;
	for (int sampleID = 0; sampleID < mpMesh->amount; sampleID++) {

		vec4 derv = mCurvatureDerivatives[sampleID];
		float v = derv[0];
		if (cml::dot(mCurvatures[sampleID].d1, vec3(1.0f, 1.0f, 1.0f)) < 0) v = -v; // HACK: consistent direction for better visualization

		meshColor.push_back(vec3i((int)(cml::clamp(v, 0.0f, 100.0f) * 255), 0, -(int)(cml::clamp(v, -100.0f, 0.0f) * 255)));
	}

	PlyExporter pe;
	if (!pe.addPoint(&mpMesh->positions, &mpMesh->normals, &meshColor)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeDirection(string fileName) {

	vector<vec3> lines;

	float r = 0.0001f; // HACK: manual radius...

	for (int sampleID = 0; sampleID < mpMesh->amount; sampleID++) {
		vec3 p = mpMesh->positions[sampleID];
		float eMax = mCurvatureDerivatives[sampleID][0];
		vec3 tMax = mCurvatures[sampleID].d1;
		vec3 tMin = mCurvatures[sampleID].d2;
		vec3 dir = cml::normalize(tMax * eMax);

		if (eMax) {
			lines.push_back(p - dir*r*10.f);
			lines.push_back(p + dir*r*50.f);
		} else {
			lines.push_back(p - tMax*r*10.f);
			lines.push_back(p + tMax*r*10.f);
		}
		lines.push_back(p - tMin*r*10.f);
		lines.push_back(p + tMin*r*10.f);
	}

	PlyExporter pe;
	if (!pe.addLine(&lines)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeBoundary(string fileName) {

	vector<vec3> pB;
	for (int edgeID : mBoundaryEdges) {
		vec3i faceIdx = mpMesh->indices[edgeID / 3];
		int v1 = faceIdx[edgeID % 3];
		int v2 = faceIdx[(edgeID + 1) % 3];
		pB.push_back(mpMesh->positions[v1]);
		pB.push_back(mpMesh->positions[v2]);
	}

	PlyExporter pe;
	if (!pe.addLine(&pB)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeCrossing(string fileName) {

	vector<vec3> lines;
	int numPoints = mCurvePoints.amount;
	for (int pointID = 0; pointID < numPoints; pointID++) {
		for (int neighborID : mCurveGraph[pointID]) {
			if (pointID < neighborID) { // add only once
				lines.push_back(mCurvePoints.positions[pointID]);
				lines.push_back(mCurvePoints.positions[neighborID]);
			}
		}
	}

	PlyExporter pe;
	if (!pe.addLine(&lines)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool CurveRidgeValley::visualizeChains(string fileName) {

	int numChains = (int)mCurveChains.size();
	vector<vec3i> chainColors(numChains);
	for (int j = 0; j < numChains; j++) {
		chainColors[j] = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
	}

	vector<vec3> vP;
	vector<vec3i> vC;
	float r = 0.001f; // HACK: visualize dense points...
	for (int chainID = 0; chainID < numChains; chainID++) {
		for (vec2i segID : mCurveChains[chainID]) {
			vec3 v1 = mCurvePoints.positions[segID[0]];
			vec3 v2 = mCurvePoints.positions[segID[1]];
			vec3 vd = cml::normalize(v2 - v1);
			float l = (v1 - v2).length();
			int m = (int)(l / r);
			float a = l / m;
			for (int j = 0; j < m; j++) {
				vP.push_back(v1 + vd*a*j);
				vC.push_back(chainColors[chainID]);
			}
		}
	}

	PlyExporter pe;
	stringstream ss;
	ss << "NUMBER OF CHAINS " << numChains;
	if (!pe.addComment(ss.str())) return false;
	if (!pe.addPoint(&vP, 0, &vC)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}