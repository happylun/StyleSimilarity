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

#include "MeshUtil.h"

#include <fstream>
#include <sstream>
#include <set>

#include "Data/StyleSimilarityConfig.h"
#include "Utility/PlyExporter.h"
#include "Utility/PlyLoader.h"

#include "Sample/SampleUtil.h"

using namespace StyleSimilarity;

bool MeshUtil::saveMesh(string fileName, TTriangleMesh &mesh) {

	PlyExporter pe;
	if (!pe.addMesh(&mesh.indices, &mesh.positions, &mesh.normals)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}

bool MeshUtil::loadMesh(string fileName, TTriangleMesh &mesh) {

	if (!PlyLoader::loadMesh(fileName, &mesh.indices, &mesh.positions, &mesh.normals)) return false;
	mesh.amount = (int)mesh.positions.size();

	return true;
}

bool MeshUtil::cleanUp(TTriangleMesh &mesh) {

	// remove zero area faces
	set<vec3i> newIndicesSet;
	vector<bool> verticesFlag(mesh.amount, false);
	for (int faceID = 0; faceID < (int)mesh.indices.size(); faceID++) {
		vec3i idx = mesh.indices[faceID];
		if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) continue;
		vec3 pos[3];
		for (int k = 0; k < 3; k++) pos[k] = mesh.positions[idx[k]];
		vec3d v1 = pos[1] - pos[0];
		vec3d v2 = pos[2] - pos[0];
		if (cml::dot(v1, v2) < 0) v2 = -v2;
		if (cml::unsigned_angle(v1, v2) < 1e-5) continue;
		newIndicesSet.insert(idx);
		for (int k = 0; k<3; k++) verticesFlag[idx[k]] = true;
	}
	vector<vec3i> newIndices(newIndicesSet.begin(), newIndicesSet.end());

	// remove unreferenced vertices
	vector<vec3> newPositions;
	vector<vec3> newNormals;
	vector<int> verticesMap(mesh.amount);
	for (int vertID = 0; vertID < mesh.amount; vertID++) {
		if (verticesFlag[vertID]) {
			verticesMap[vertID] = (int)newPositions.size();
			newPositions.push_back(mesh.positions[vertID]);
			newNormals.push_back(mesh.normals[vertID]);
		}
	}

	// update face indices
	for (vec3i &idx : newIndices) {
		for (int k = 0; k < 3; k++) idx[k] = verticesMap[idx[k]];
	}


	mesh.positions.swap(newPositions);
	mesh.normals.swap(newNormals);
	mesh.indices.swap(newIndices);
	mesh.amount = (int)mesh.positions.size();

	return true;
}

bool MeshUtil::subdivideMesh(
	TTriangleMesh &inMesh,
	TTriangleMesh &outMesh,
	vector<int> &outFaceIndices,
	double radius)
{

	// clone existing vertices
	outMesh.positions.assign(inMesh.positions.begin(), inMesh.positions.end());
	outMesh.normals.assign(inMesh.normals.begin(), inMesh.normals.end());

	// compute face area
	int numFaces = (int)inMesh.indices.size();

	float maxLength;
	if (radius) {
		maxLength = (float)(radius * 2);
	} else {
		// compute radius from default configuration
		float totalArea = 0;
		for (int faceID = 0; faceID < numFaces; faceID++) {
			vec3i faceIdx = inMesh.indices[faceID];
			vec3 facePos[3];
			for (int j = 0; j < 3; j++) facePos[j] = inMesh.positions[faceIdx[j]];
			float area = cml::cross(facePos[1] - facePos[0], facePos[2] - facePos[0]).length();
			totalArea += area;
		}
		int numSamples = StyleSimilarityConfig::mSample_WholeMeshSampleNumber;
		maxLength = sqrt(totalArea / (sqrt(3.0f)*numSamples / 2)) * 0.76f * 2;
	}

	// process each triangle
	outMesh.indices.clear();
	outFaceIndices.clear();
	for (int faceID = 0; faceID < numFaces; faceID++) {

		vec3i faceIdx = inMesh.indices[faceID];
		vec3 facePos[3];
		for (int j = 0; j < 3; j++) facePos[j] = inMesh.positions[faceIdx[j]];
		float length = 0;
		for (int j = 0; j < 3; j++) length = max(length, (facePos[(j + 1) % 3] - facePos[j]).length());

		if(length > maxLength) {
			// subdivide long triangle
			int div = (int)ceil(length / maxLength);

			// add new vertices
			vector<vector<int>> divPointIdx(div+1, vector<int>(div+1));
			for (int j = 0; j <= div; j++) {
				for (int v = 0; v <= j; v++) {
					int u = j - v;
					if (u == 0 && v == 0) divPointIdx[u][v] = faceIdx[0];
					else if (u == div && v == 0) divPointIdx[u][v] = faceIdx[1];
					else if (u == 0 && v == div) divPointIdx[u][v] = faceIdx[2];
					else {
						divPointIdx[u][v] = (int)outMesh.positions.size();						
						outMesh.positions.push_back(facePos[0] + ((facePos[1] - facePos[0]) * u + (facePos[2] - facePos[0]) * v) / div);
						outMesh.normals.push_back(vec3()); // will calculate later
					}

				}
			}

			// add new faces
			for (int j = 0; j <= div; j++) {
				for (int v = 0; v <= j; v++) {
					int u = j - v;
					if (j < div) {
						outMesh.indices.push_back(vec3i(
							divPointIdx[u][v],
							divPointIdx[u+1][v],
							divPointIdx[u][v+1]));
						outFaceIndices.push_back(faceID);
						if (u == 0) continue;
						outMesh.indices.push_back(vec3i(
							divPointIdx[u][v],
							divPointIdx[u][v+1],
							divPointIdx[u-1][v+1]));
						outFaceIndices.push_back(faceID);
					}
				}
			}
		} else {
			if (length == 0) continue; // skip zero area faces
			// retain original triangle
			outMesh.indices.push_back(faceIdx);
			outFaceIndices.push_back(faceID);
		}
	}
	outMesh.amount = (int)outMesh.positions.size();

	// compute normal
	if (!recomputeNormals(outMesh)) return false;

	return true;
}

bool MeshUtil::subdivideMeshKeepTopology(
	TTriangleMesh &inMesh,
	TTriangleMesh &outMesh,
	vector<int> &outFaceIndices,
	double radius)
{

	// clone existing vertices
	outMesh.positions.assign(inMesh.positions.begin(), inMesh.positions.end());
	outMesh.normals.assign(inMesh.normals.begin(), inMesh.normals.end());

	int numFaces = (int)inMesh.indices.size();

	// compute max length of edge

	float maxLength;
	if (radius) {
		maxLength = (float)(radius * 2);
	} else {
		// compute radius from default configuration
		float totalArea = 0;
		for (int faceID = 0; faceID < numFaces; faceID++) {
			vec3i faceIdx = inMesh.indices[faceID];
			vec3 facePos[3];
			for (int j = 0; j < 3; j++) facePos[j] = inMesh.positions[faceIdx[j]];
			float area = cml::cross(facePos[1] - facePos[0], facePos[2] - facePos[0]).length();
			totalArea += area;
		}
		int numSamples = StyleSimilarityConfig::mSample_WholeMeshSampleNumber;
		maxLength = sqrt(totalArea / (sqrt(3.0f)*numSamples / 2)) * 0.76f;
	}

	// divide each edge (add new vertices)

	vector<vector<int>> edgePointList; // point ID : # points on edge : # of edges
	map<vec2i, int> edgeMap; // edge ID : point ID pair (small first)

	for (int faceID = 0; faceID < numFaces; faceID++) {

		vec3i faceIdx = inMesh.indices[faceID];
		for (int j = 0; j < 3; j++) {
			int pID1 = faceIdx[j];
			int pID2 = faceIdx[(j + 1) % 3];
			if (pID1 > pID2) swap(pID1, pID2);
			vec2i edgeKey(pID1, pID2);
			auto it = edgeMap.find(edgeKey);
			if (it == edgeMap.end()) {
				vec3 p1 = inMesh.positions[pID1];
				vec3 n1 = inMesh.normals[pID1];
				vec3 p2 = inMesh.positions[pID2];				
				vec3 n2 = inMesh.normals[pID2];
				float length = (p2 - p1).length();
				vector<int> pointList;
				pointList.push_back(pID1);
				if (length > maxLength) {
					// divide edge (add divide points)
					int div = (int)(ceil(length / maxLength));					
					for (int k = 1; k < div; k++) {
						float r = k / (float)div;
						vec3 pNew = p2*r + p1*(1 - r);
						vec3 nNew = cml::normalize(n2*r + n1*(1 - r));
						int iNew = (int)outMesh.positions.size();
						outMesh.positions.push_back(pNew);
						outMesh.normals.push_back(nNew);
						pointList.push_back(iNew);
					}					
				}
				pointList.push_back(pID2);
				edgeMap[edgeKey] = (int)edgePointList.size();
				edgePointList.push_back(pointList);
			}
		}
	}

	// first pass: divide triangle along one edge (the one to the vertex with smallest ID)

	vector<vec3i> tmpFaces(0);
	vector<int> tmpFaceIndices(0);

	for (int faceID = 0; faceID < numFaces; faceID++) {

		vec3i faceIdx = inMesh.indices[faceID];

		// rotate indices to make first index smallest
		if (faceIdx[1] <= faceIdx[0] && faceIdx[1] <= faceIdx[2]) {
			faceIdx = vec3i(faceIdx[1], faceIdx[2], faceIdx[0]);
		} else if (faceIdx[2] <= faceIdx[0] && faceIdx[2] <= faceIdx[1]) {
			faceIdx = vec3i(faceIdx[2], faceIdx[0], faceIdx[1]);
		}

		int topPointID = faceIdx[0];
		vec2i bottomEdgeKey = vec2i(faceIdx[1], faceIdx[2]);
		bool isSwapped = false;
		if (bottomEdgeKey[0] > bottomEdgeKey[1]) {
			swap(bottomEdgeKey[0], bottomEdgeKey[1]);
			isSwapped = true;
		}
		int bottomEdgeID = edgeMap[bottomEdgeKey];
		auto bottomEdgePointList = edgePointList[bottomEdgeID];
		for (int k = 0; k < (int)bottomEdgePointList.size() - 1; k++) {
			if (k > 0) {
				// add internal edge
				int pID1 = topPointID;
				int pID2 = bottomEdgePointList[k];
				vec2i edgeKey(pID1, pID2);
				vec3 p1 = outMesh.positions[pID1];
				vec3 n1 = outMesh.normals[pID1];
				vec3 p2 = outMesh.positions[pID2];
				vec3 n2 = outMesh.normals[pID2];
				float length = (p2 - p1).length();
				vector<int> pointList;
				pointList.push_back(pID1);
				if (length > maxLength) {
					// divide edge (add divide points)
					int div = (int)(ceil(length / maxLength));
					for (int k = 1; k < div; k++) {
						float r = k / (float)div;
						vec3 pNew = p2*r + p1*(1 - r);
						vec3 nNew = cml::normalize(n2*r + n1*(1 - r));
						int iNew = (int)outMesh.positions.size();
						outMesh.positions.push_back(pNew);
						outMesh.normals.push_back(nNew);
						pointList.push_back(iNew);
					}
				}
				pointList.push_back(pID2);
				edgeMap[edgeKey] = (int)edgePointList.size();
				edgePointList.push_back(pointList);
			}
			vec3i face = vec3i(topPointID, bottomEdgePointList[k], bottomEdgePointList[k + 1]);
			if (isSwapped) swap(face[1], face[2]);
			tmpFaces.push_back(face);
			tmpFaceIndices.push_back(faceID);
		}
	}

	// second pass: divide triangle along two divided edge

	outMesh.indices.clear();
	outFaceIndices.clear();

	for (int faceID = 0; faceID < (int)tmpFaces.size(); faceID++) {

		vec3i faceIdx = tmpFaces[faceID];
		int edgeID1 = edgeMap[vec2i(faceIdx[0], faceIdx[1])];
		int edgeID2 = edgeMap[vec2i(faceIdx[0], faceIdx[2])];
		auto &pointList1 = edgePointList[edgeID1];
		auto &pointList2 = edgePointList[edgeID2];
		if (pointList1[0] != faceIdx[0] || pointList2[0] != faceIdx[0]) {
			cout << "Error: incorrect point list" << endl;
			return false;
		}
		int n1 = (int)pointList1.size();
		int n2 = (int)pointList2.size();
		int p1 = 1;
		int p2 = 1;
		float l1 = (outMesh.positions[pointList1[p1]] - outMesh.positions[pointList1[0]]).length_squared();
		float l2 = (outMesh.positions[pointList2[p2]] - outMesh.positions[pointList2[0]]).length_squared();
		outMesh.indices.push_back(vec3i(faceIdx[0], pointList1[p1], pointList2[p2]));
		outFaceIndices.push_back(tmpFaceIndices[faceID]);
		while (true) {
			int oldP1 = p1;
			int oldP2 = p2;
			if (p1 == n1 - 1) p2++;
			else if (p2 == n2 - 1) p1++;
			else if (l1 < l2) p1++;
			else p2++;
			if (p1 >= n1 || p2 >= n2) {
				break;
			}
			if (oldP1 != p1) {
				l1 = (outMesh.positions[pointList1[p1]] - outMesh.positions[pointList1[0]]).length_squared();
				outMesh.indices.push_back(vec3i(pointList1[oldP1], pointList1[p1], pointList2[oldP2]));
				outFaceIndices.push_back(tmpFaceIndices[faceID]);
			} else {
				l2 = (outMesh.positions[pointList2[p2]] - outMesh.positions[pointList2[0]]).length_squared();
				outMesh.indices.push_back(vec3i(pointList1[oldP1], pointList2[p2], pointList2[oldP2]));
				outFaceIndices.push_back(tmpFaceIndices[faceID]);
			}
		}
	}

	outMesh.amount = (int)outMesh.positions.size();

	// compute normal
	if (!recomputeNormals(outMesh)) return false;

	return true;
}

bool MeshUtil::extractMeshFromSamplePoints(
	TTriangleMesh &inMesh, // subdivided mesh
	TSampleSet &inSamples, // all sample points
	vector<bool> &inFlags, // valid sample flag : # sample points
	TTriangleMesh &outMesh) // subset of inMesh
{
	int numFaces = (int)inMesh.indices.size();

	// build tree
	SKDTree tree;
	SKDTreeData treeData;
	if (!SampleUtil::buildKdTree(inSamples.positions, tree, treeData)) return false;
	
	// build query point sets (center of faces)
	Eigen::Matrix3Xd faceCenters(3, numFaces);
	for (int faceID = 0; faceID < numFaces; faceID++) {
		vec3i idx = inMesh.indices[faceID];
		vec3 center = (inMesh.positions[idx[0]] + inMesh.positions[idx[1]] + inMesh.positions[idx[2]]) / 3;
		faceCenters.col(faceID) = Eigen::Vector3d(center[0], center[1], center[2]);
	}

	// create sub-mesh
	Eigen::VectorXi faceSamples;
	if (!SampleUtil::findNearestNeighbors(tree, faceCenters, faceSamples)) return false;
	outMesh = inMesh;
	outMesh.indices.clear();
	for (int faceID = 0; faceID < numFaces; faceID++) {
		if (inFlags[faceSamples[faceID]]) outMesh.indices.push_back(inMesh.indices[faceID]);
	}

	return true;
}

bool MeshUtil::recomputeNormals(TTriangleMesh &mesh) {

	int numVertices = mesh.amount;

	vector<vec3d> normals(numVertices, vec3d(0.0,0.0,0.0));
	vector<vec3d> binormals(numVertices, vec3d(0.0,0.0,0.0));
	vector<vec3d> rawnormals(numVertices, vec3d(0.0,0.0,0.0));
	vector<double> weights(numVertices, 0.0);

	for(vec3i faceIdx : mesh.indices) {
		vec3d v[3];
		for(int j=0; j<3; j++) v[j] = mesh.positions[faceIdx[j]];
		vec3d vv1 = cml::normalize(v[1] - v[0]);
		vec3d vv2 = cml::normalize(v[2] - v[0]);
		vec3d n = cml::cross(vv1, vv2);
		if (n.length_squared()) n.normalize();
		double w = cml::unsigned_angle(vv1, vv2); // use angle as weight to alleviate precision issue (when weighted by area)
		vec3d bn[3];
		for(int j=0; j<3; j++) bn[j] = cml::normalize( v[j] - (v[(j+1)%3] + v[(j+2)%3])/2 );
		for(int j=0; j<3; j++) {
			if (faceIdx[j] == 5862) {
				int k = 0;
			}
			rawnormals[faceIdx[j]] += n;
			normals[faceIdx[j]] += n * w;
			binormals[faceIdx[j]] += bn[j] * w;
			weights[faceIdx[j]] += w;
		}
	}

	bool warnFlag = false;
	for(int j=0; j<numVertices; j++) {
		if (normals[j].length_squared() > 0) {
			mesh.normals[j] = (vec3)cml::normalize(normals[j]);
		} else if (binormals[j].length_squared() > 0) {
			mesh.normals[j] = (vec3)cml::normalize(binormals[j]);
		} else if (rawnormals[j].length_squared() > 0) {
			mesh.normals[j] = (vec3)cml::normalize(rawnormals[j]);
		} else {
			if (!warnFlag) {
				cout << "Warning: detected zero area faces or unreferenced vertices" << endl;
				warnFlag = true;
			}
			mesh.normals[j] = vec3(0.0f, 1.0f, 0.0f);
		}
	}

	return true;
}

bool MeshUtil::computeDihedralAngle(vec3 center1, vec3 normal1, vec3 center2, vec3 normal2, double &angle) {

	double dihedralAngle = cml::constantsd::pi() - cml::acos_safe((double)cml::dot(normal1, normal2));

	vec3 centerDir = center1 - center2;
	float cosAngle1 = cml::dot(centerDir, normal2); // unnormalized
	float cosAngle2 = cml::dot(-centerDir, normal1); // unnormalized
	bool flag1 = (cosAngle1 >= 0);
	bool flag2 = (cosAngle2 >= 0);

	if (flag1 && flag2) {
		// concave
	} else if (!flag1 && !flag2) {
		// convex
		dihedralAngle = cml::constantsd::two_pi() - dihedralAngle;
	} else {
		if (cml::dot(normal1, normal2) > 0.9f) {
			// prevent numerical error on flat plane
		} else {
			// face orientation incompatible
			dihedralAngle = -cml::constantsd::pi();
		}
	}

	angle = dihedralAngle;

	return true;
}

bool MeshUtil::computeAABB(TTriangleMesh &mesh, vec3 &bbMin, vec3 &bbMax) {

	bbMin = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	bbMax = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (vec3i idx : mesh.indices) {
		for (int j = 0; j < 3; j++) {
			bbMin.minimize(mesh.positions[idx[j]]);
			bbMax.maximize(mesh.positions[idx[j]]);
		}
	}

	return true;
}
