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

#include "SamplePoissonDisk.h"

#include <set>

#include "Utility/PlyExporter.h"
#include "Utility/PlyLoader.h"

#include "Mesh/MeshUtil.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

const double SamplePoissonDisk::RELATIVE_RADIUS = 0.76;

#define DEBUG_OUTPUT

#define USE_INTERPOLATED_NORMAL  1
#define USE_FACE_NORMAL          2
//#define USE_WHICH_NORMAL         USE_INTERPOLATED_NORMAL
#define USE_WHICH_NORMAL         USE_FACE_NORMAL

SamplePoissonDisk::SamplePoissonDisk(TTriangleMesh *mesh) {

	clearUp();

	mpMesh = mesh;

	if (!MeshUtil::computeAABB(*mpMesh, mBBMin, mBBMax)) cout << "Error: mesh AABB" << endl;
	if (!MeshUtil::subdivideMesh(*mpMesh, mDenseMesh, mDenseMeshFaceIndices)) cout << "Error: subdivide mesh" << endl;
	if (!MeshUtil::cleanUp(mDenseMesh)) cout << "Error: clean up dense mesh" << endl;

	mTriangleFlags.clear();
	mTriangleFlags.resize(mDenseMesh.indices.size(), true);

	mSampleRadius = (mBBMax - mBBMin).length() * 1e-5; // temporary radius
}

SamplePoissonDisk::~SamplePoissonDisk() {

	clearUp();
}

bool SamplePoissonDisk::runSampling(int numSamples) {

	if (StyleSimilarityConfig::mSample_VisibilityChecking) {
		if (!buildKDTree()) return false;
		if (!pruneInteriorFaces()) return false;
	}

	long long n = numSamples; // be careful! may be very large

	while( (double)mSamples.size() < numSamples * StyleSimilarityConfig::mSample_MinimumSampleRate) {
		// while not enough samples, redo sampling

		if (!calculateCDF()) return false;
		if( !buildGrids( (int)n ) ) return false;
		if( !generateSamples( numSamples ) ) return false;
		n = n * numSamples / max(1, (int)mSamples.size());
#ifdef DEBUG_OUTPUT
		cout << "numSamples: " << mSamples.size() << endl;
#endif
	}

	return true;
}

bool SamplePoissonDisk::buildKDTree() {

#ifdef DEBUG_OUTPUT
	cout << "Building KD Tree..." << endl;
#endif

	mMeshTreeData.resize(mpMesh->indices.size());
#pragma omp parallel for
	for(int faceID=0; faceID<(int)mpMesh->indices.size(); faceID++) {
		vec3i idx = mpMesh->indices[faceID];
		G3D::Vector3 v0(mpMesh->positions[idx[0]].data());
		G3D::Vector3 v1(mpMesh->positions[idx[1]].data());
		G3D::Vector3 v2(mpMesh->positions[idx[2]].data());
		TKDTreeElement tri(TKDT::NamedTriangle(v0, v1, v2, faceID));
		mMeshTreeData[faceID] = tri;
	}

	if (StyleSimilarityConfig::mSample_AddVirtualGround) {

		if (mpMesh->amount == 0) {
			cout << "Error: empty mesh" << endl;
			return false;
		}

		float r = (float)mSampleRadius;
		G3D::Vector3 v0(mBBMin[0], mBBMin[1] - r, mBBMin[2]);
		G3D::Vector3 v1(mBBMax[0], mBBMin[1] - r, mBBMin[2]);
		G3D::Vector3 v2(mBBMax[0], mBBMin[1] - r, mBBMax[2]);
		G3D::Vector3 v3(mBBMin[0], mBBMin[1] - r, mBBMax[2]);
		TKDTreeElement f0(TKDT::NamedTriangle(v0, v1, v2, -1));
		TKDTreeElement f1(TKDT::NamedTriangle(v0, v2, v3, -1));
		mMeshTreeData.push_back(f0);
		mMeshTreeData.push_back(f1);
	}

	mMeshTree.init(mMeshTreeData.begin(), mMeshTreeData.end());

	return true;
}

bool SamplePoissonDisk::pruneInteriorFaces() {

	int numFaces = (int)mDenseMesh.indices.size();
	if (numFaces > StyleSimilarityConfig::mSample_MaximumCheckedFaceCount ||
		StyleSimilarityConfig::mSample_MaximumCheckedFaceCount == 0) {
		return true; // skip pre-checking
	}

#ifdef DEBUG_OUTPUT
	cout << "Pruning interior faces..." << endl;
#endif

	const int numSamples = 16;

	// test each face
//#pragma omp parallel for
	for(int faceID=0; faceID<numFaces; faceID++) {
		bool valid = false;
		for (int sampleID = 0; sampleID < numSamples; sampleID++) {
			TPoissonSample sample;
			if (!sampleOnTriangle(sample, faceID)) {
				cout << "Error: sample on triangle" << endl;
				break;
			}
			if (checkVisibility(sample)) {
				valid = true;
				break;
			}
		}

		if(!valid) mTriangleFlags[faceID] = false;

#ifdef DEBUG_OUTPUT
		if(faceID%1000==999) cout << "\rChecked " << (faceID+1) << " / " << numFaces << "            ";
#endif
	}
#ifdef DEBUG_OUTPUT
	cout << "\rChecked " << numFaces << " / " << numFaces << endl;
#endif

	return true;
}

bool SamplePoissonDisk::calculateCDF() {

#ifdef DEBUG_OUTPUT
	cout << "Calculating CDF..." << endl;
#endif

	int numFaces = (int)mDenseMesh.indices.size();

	// compute triangle area
	vector<double> faceArea(numFaces, 0);
#pragma omp parallel for
	for(int faceID=0; faceID<numFaces; faceID++) {
		if(mTriangleFlags[faceID]) {
			vec3i &idx = mDenseMesh.indices[faceID];
			vec3d v[3];
			for(int j=0; j<3; j++) v[j] = mDenseMesh.positions[idx[j]];
			faceArea[faceID] = cml::cross((v[1]-v[0]), (v[2]-v[0])).length() / 2;
		}
	}

	mTotalArea = 0;
	mTriangleAreaCDF.clear();
	for (int faceID = 0; faceID < numFaces; faceID++) {
		mTotalArea += faceArea[faceID];
		mTriangleAreaCDF.push_back(mTotalArea);
	}

	if(mTotalArea == 0) {
		cout << "Error: empty mesh" << endl;
		return false;
	}

	// normalize CDF
	double denom = 1.0 / mTotalArea;
	for(double &it : mTriangleAreaCDF) {
		it *= denom;
	}

	return true;
}

bool SamplePoissonDisk::buildGrids(int numSamples) {

#ifdef DEBUG_OUTPUT
	cout << "Building grids..." << endl;
#endif

	mSampleRadius = sqrt(mTotalArea / (sqrt(3.0)*numSamples/2)) * RELATIVE_RADIUS;

	// calculate grid size

	const int maxGirdSize = 500; // UNDONE: param maximum grid dimension

	vec3 bbSize = mBBMax - mBBMin;
	double maxBBSize = (double) max(bbSize[0], max(bbSize[1], bbSize[2]));

	mGridRadius = max(mSampleRadius, maxBBSize/maxGirdSize);
	mGridSize[0] = cml::clamp((int)ceil( (double)bbSize[0] / mGridRadius ), 1, maxGirdSize);
	mGridSize[1] = cml::clamp((int)ceil( (double)bbSize[1] / mGridRadius ), 1, maxGirdSize);
	mGridSize[2] = cml::clamp((int)ceil( (double)bbSize[2] / mGridRadius ), 1, maxGirdSize);

	// init grids

	long long gridAllSize = mGridSize[0]*mGridSize[1]*mGridSize[2];
	if(gridAllSize > 200000000LL) {
		// requires too much memory, infeasible
		mSamples.clear();
		cout << "Error: requires too many grids" << endl;
		return false;
	}

	mGrids.clear();
	mGrids.resize((int)gridAllSize, vector<int>(0));

	return true;
}

bool SamplePoissonDisk::generateSamples(int numSamples) {

#ifdef DEBUG_OUTPUT
	cout << "Start sampling..." << endl;
#endif

	// Poisson disk sampling

	int numFaces = (int)mDenseMesh.indices.size();

	mSamples.clear();
	mSamples.resize(numSamples);
	int failCount = 0;

	vector<int> faceFailCounts(numFaces, 0);
	vector<int> faceHitCounts(numFaces, 0);

	int sampleID;
	for(sampleID=0; sampleID<numSamples;) {

		// generate samples

		int faceID;
		if( !chooseTriangle(faceID) ) return false;

		TPoissonSample sample;
		if( !sampleOnTriangle(sample, faceID) ) return false;

		// check if sample is valid

		bool valid = true;

		int gridPos;
		if( !checkGrid(sample, gridPos) ) {
			valid = false;
		} else {
			if (!checkVisibility(sample)) {
				valid = false;
				faceFailCounts[faceID]++;
			}
			faceHitCounts[faceID]++;
		}

		// accept or discard sample

		if(valid) {
#ifdef DEBUG_OUTPUT
			if(sampleID%1000 == 999) cout << "\rSampling: " << (sampleID+1) << " / " << numSamples << " : " << failCount << "        ";
#endif

			mGrids[gridPos].push_back(sampleID);
			mSamples[sampleID] = sample;
			failCount = 0;
			sampleID++;
		} else {
			failCount++;
			if(failCount > StyleSimilarityConfig::mSample_MaximumFailedCount) break;
		}
	}
#ifdef DEBUG_OUTPUT
	cout << "\rSampling: " << (sampleID + 1) << " / " << numSamples << " : " << failCount << endl;
#endif

	mSamples.resize(sampleID);

	// update face flag
#pragma omp parallel for
	for (int faceID = 0; faceID < numFaces; faceID++) {
		if (faceHitCounts[faceID] > 10 && faceHitCounts[faceID] == faceFailCounts[faceID]) { // UNDONE: param
			mTriangleFlags[faceID] = false;
		}
	}

	return true;
}

bool SamplePoissonDisk::chooseTriangle(int &faceID) {

	// choose a triangle based on CDF

	int numFaces = (int)mDenseMesh.indices.size();

	double rng = getPreciseRandomNumber();
	// binary search
	int p1 = -1;
	int p2 = numFaces-1;
	while(p1+1 < p2) {
		int pm = (p1+p2)/2;
		if(mTriangleAreaCDF[pm] < rng) {
			p1 = pm;
		} else {
			p2 = pm;
		}
	}

	faceID = p2;
	if (rng > 0) {
		// choose first valid face (leftmost among faces with the same CDF)
		while (faceID > 0 && mTriangleAreaCDF[faceID] == mTriangleAreaCDF[faceID - 1]) faceID--;
	} else {
		// choose first valid face (first with non-zero CDF)
		while (faceID < numFaces - 1 && mTriangleAreaCDF[faceID] == 0) faceID++;
	}
	if (!mTriangleFlags[faceID]) {
		cout << "Error: sampled on interior face " << faceID << endl;
		return false;
	}

	return true;
}

bool SamplePoissonDisk::sampleOnTriangle(TPoissonSample &sample, int faceID) {

	// sample within a triangle

	vec3i idx = mDenseMesh.indices[faceID];
	vec3 v[3], n[3];
	for(int j=0; j<3; j++) {
		v[j] = mDenseMesh.positions[idx[j]];
		n[j] = mDenseMesh.normals[idx[j]];
	}

	double rng1 = cml::random_unit();
	double rng2 = cml::random_unit();
	float r1 = (float)(1 - sqrt(rng1));
	float r2 = (float)(rng2 * sqrt(rng1));

	sample.position = v[0] + (v[1]-v[0]) * r1 + (v[2]-v[0]) * r2;
#if USE_WHICH_NORMAL == USE_INTERPOLATED_NORMAL
	// be careful ! vertex normal is not reliable
	sample.normal = cml::normalize( n[0] + (n[1]-n[0]) * r1 + (n[2]-n[0]) * r2 );
#elif USE_WHICH_NORMAL == USE_FACE_NORMAL
	sample.normal = cml::cross(v[1]-v[0], v[2]-v[0]).normalize();
#else
	#error normal method not specified
#endif
	sample.faceID = mDenseMeshFaceIndices[faceID];

	return true;
}

bool SamplePoissonDisk::checkGrid(TPoissonSample &sample, int &gridPos) {

	// check grid
	bool valid = true;

	vec3 relPos = sample.position - mBBMin;
	int gridX = cml::clamp( (int)floor( (double)relPos[0] / mGridRadius ), 0, mGridSize[0]-1 );
	int gridY = cml::clamp( (int)floor( (double)relPos[1] / mGridRadius ), 0, mGridSize[1]-1 );
	int gridZ = cml::clamp( (int)floor( (double)relPos[2] / mGridRadius ), 0, mGridSize[2]-1 );
	gridPos = (gridX * mGridSize[1] + gridY) * mGridSize[2] + gridZ;

	// check within 3x3x3 neighbor cells
	for(int offsetX=-1; offsetX<=1; offsetX++) {
		int clampX = cml::clamp(gridX+offsetX, 0, mGridSize[0]-1);
		if(gridX+offsetX != clampX) continue;
		for(int offsetY=-1; offsetY<=1; offsetY++) {
			int clampY = cml::clamp(gridY+offsetY, 0, mGridSize[1]-1);
			if(gridY+offsetY != clampY) continue;
			for(int offsetZ=-1; offsetZ<=1; offsetZ++) {
				int clampZ = cml::clamp(gridZ+offsetZ, 0, mGridSize[2]-1);
				if(gridZ+offsetZ != clampZ) continue;

				int checkPos = (clampX * mGridSize[1] + clampY) * mGridSize[2] + clampZ;
				for(int nbID : mGrids[checkPos]) {
					TPoissonSample &neighbor = mSamples[nbID];
					if(cml::dot(sample.normal, neighbor.normal) > 0) { // normal compatible

						vec3 &p1 = sample.position;
						vec3 &n1 = sample.normal;
						vec3 &p2 = neighbor.position;
						vec3 &n2 = neighbor.normal;
						vec3 v = p2-p1;

						// estimate geodesic distance
						//float dist = v.length();
						//v.normalize();
						//float c1 = cml::dot(n1, v);
						//float c2 = cml::dot(n2, v);
						//dist *= fabs(c1-c2) < 1e-7 ? (1.0f/sqrt(1.0f-c1*c1)) : (asin(c1)-asin(c2))/(c1-c2);

						// use L1 distance
						//float dist = fabs(v[0]) + fabs(v[1]) + fabs(v[2]);

						// use Euclid distance
						float dist = v.length();

						if( dist < mSampleRadius ) {
							valid = false;
							break;
						}
					}
				}
				if(!valid) break;
			}
			if(!valid) break;
		}
		if(!valid) break;
	}

	return valid;
}

bool SamplePoissonDisk::checkVisibility(TPoissonSample &sample) {

	if (!StyleSimilarityConfig::mSample_VisibilityChecking) return true; // skip visibility checking

	const int numRays = 16; // UNDONE: param number of rays for visibility checking
	float eps = max((float)(mSampleRadius * 0.01), 1e-5f); // radius may not be specified. use 1e-5 as minimum

	int unhitCount = 0;
#pragma omp parallel for shared(unhitCount)
	for(int rayID=0; rayID<numRays; rayID++) {

		if (unhitCount > 0) continue;

		vec3 rayDir;
		while (true) {
			double r1 = cml::random_unit();
			double r2 = cml::random_unit();
			rayDir = vec3d(
				2.0 * cos(cml::constantsd::two_pi()*r1) * cml::sqrt_safe(r2*(1 - r2)),
				2.0 * sin(cml::constantsd::two_pi()*r1) * cml::sqrt_safe(r2*(1 - r2)),
				1.0 - 2.0*r2); // random direction on unit sphere
			float cosA = cml::dot(rayDir, sample.normal);
			if (fabs(cosA) > 0.5f) { // within 60 degree
				if (cosA < 0) {
					rayDir = -rayDir; // flip if in opposite direction
				}
				break;
			}
		}

		vec3 rayOrigin = sample.position + rayDir * eps; // add eps for offset
		Thea::Ray3 ray( G3D::Vector3(rayOrigin.data()), G3D::Vector3(rayDir.data()) );
		float dist = (float)mMeshTree.rayIntersectionTime(ray);

		if(dist<-0.5) { // should be -1 if not hit
#pragma omp atomic
			unhitCount++;
		}
	}

	return unhitCount > 0;
}

bool SamplePoissonDisk::exportSample(TSampleSet &samples) {

	samples.positions.clear();
	samples.normals.clear();
	samples.indices.clear();

	for(auto &sample : mSamples) {
		samples.positions.push_back( sample.position );
		samples.normals.push_back( sample.normal );
		samples.indices.push_back( sample.faceID );
	}
	samples.amount = (int)samples.positions.size();
	samples.radius = (float)mSampleRadius;

	return true;
}

bool SamplePoissonDisk::exportDenseMesh(TTriangleMesh &mesh) {

	mesh.amount = mDenseMesh.amount;
	mesh.positions.assign(mDenseMesh.positions.begin(), mDenseMesh.positions.end());
	mesh.normals.assign(mDenseMesh.normals.begin(), mDenseMesh.normals.end());
	mesh.indices.assign(mDenseMesh.indices.begin(), mDenseMesh.indices.end());

	return true;
}

void SamplePoissonDisk::clearUp() {

	mTriangleAreaCDF.clear();
	mGrids.clear();
	mSamples.clear();
	mTriangleFlags.clear();
}

double SamplePoissonDisk::getPreciseRandomNumber() {

	long long rng = rand()*(RAND_MAX+1)+rand();
	return double(rng) / ((RAND_MAX+1)*(RAND_MAX+1));
}