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

#include "FeatureAmbientOcclusion.h"

#include <fstream>
#include <iostream>

#include "Utility/PlyExporter.h"

#include "Data/StyleSimilarityConfig.h"

#include "Feature/FeatureUtil.h"

using namespace StyleSimilarity;

FeatureAmbientOcclusion::FeatureAmbientOcclusion(TSampleSet *samples, TTriangleMesh *mesh, vector<double> *features) {

	mpSamples = samples;
	mpMesh = mesh;
	mpFeatures = features;
}

FeatureAmbientOcclusion::~FeatureAmbientOcclusion() {
}

bool FeatureAmbientOcclusion::calculate() {
	
	if (!buildKDTree()) return false;
	if (!rayCasting()) return false;

	return true;
}

bool FeatureAmbientOcclusion::buildKDTree() {
	
	mTreeData.resize(mpMesh->indices.size());

#pragma omp parallel for
	for (int faceID = 0; faceID<(int)mpMesh->indices.size(); faceID++) {
		vec3i idx = mpMesh->indices[faceID];
		G3D::Vector3 v0(mpMesh->positions[idx[0]].data());
		G3D::Vector3 v1(mpMesh->positions[idx[1]].data());
		G3D::Vector3 v2(mpMesh->positions[idx[2]].data());
		mTreeData[faceID].set(TKDT::NamedTriangle(v0, v1, v2, faceID));
	}

	mTree.init(mTreeData.begin(), mTreeData.end());

	return true;
}

bool FeatureAmbientOcclusion::rayCasting() {

	int numRays = 30;
	float eps = mpSamples->radius * 0.001f;

	mpFeatures->resize(mpSamples->amount);
#pragma omp parallel for
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {

		vec3 p = mpSamples->positions[sampleID];
		vec3 n = mpSamples->normals[sampleID];

		// count occlusion percentage
		int occludedCount = 0;
		for (int rayID = 0; rayID < numRays; rayID++) {

			// random sampling on ray direction within hemisphere along point normal
			vec3 rayDir;
			if (true) {
				double r1 = cml::random_unit();
				double r2 = cml::random_unit();
				rayDir = vec3d(
					2.0 * cos(cml::constantsd::two_pi()*r1) * cml::sqrt_safe(r2*(1 - r2)),
					2.0 * sin(cml::constantsd::two_pi()*r1) * cml::sqrt_safe(r2*(1 - r2)),
					1.0 - 2.0*r2); // random direction on unit sphere
				if (cml::dot(rayDir, n) < 0) rayDir = -rayDir; // along normal
			}

			// calculate ray intersection
			vec3 rayOrigin = p + rayDir * eps; // add eps for offset
			Thea::Ray3 ray(G3D::Vector3(rayOrigin.data()), G3D::Vector3(rayDir.data()));
			auto hitResult = mTree.rayStructureIntersection(ray);
			double dist = hitResult.getTime();
			if (dist > 0) { // intersects with mesh
				occludedCount++;
			}
		}
		
		(*mpFeatures)[sampleID] = 1.0 - occludedCount / (double)numRays;
	}

	return true;
}

bool FeatureAmbientOcclusion::visualize(string fileName) {

	PlyExporter pe;

	vector<vec3i> vColors;
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		double v = (*mpFeatures)[sampleID];
		int c = (int)(v * 255);
		vColors.push_back(vec3i(c, c, c));
	}

	if (!pe.addPoint(&mpSamples->positions, &mpSamples->normals, &vColors)) return false;
	if (!pe.output(fileName)) return false;

	return true;
}
