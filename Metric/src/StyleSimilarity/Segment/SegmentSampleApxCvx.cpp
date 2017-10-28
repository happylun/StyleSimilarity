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

#include "SegmentSampleApxCvx.h"

#include <fstream>
#include <set>

#include "Utility/PlyExporter.h"

#include "Sample/SampleUtil.h"
#include "Segment/SegmentUtil.h"
#include "Feature/FeatureUtil.h"
#include "Feature/FeatureSDF.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

#define OUTPUT_PROGRESS

SegmentSampleApxCvx::SegmentSampleApxCvx(TSampleSet *samples, TTriangleMesh *mesh) {

	mpSamples = samples;
	mpMesh = mesh;
	mSegments.clear();
	mSegmentsOffset.clear();

	mPatches.clear();
	mPatchNeighbors.clear();
	mVisibility.clear();
}

SegmentSampleApxCvx::~SegmentSampleApxCvx() {
}

bool SegmentSampleApxCvx::loadPatches(vector<vector<int>> patches, vector<vector<int>> patchGraph) {

	mPatches = patches;
	mPatchNeighbors = patchGraph;

	return true;
}

bool SegmentSampleApxCvx::runSegmentation() {

	if (mPatches.empty()) {
		cout << "Error: laod patches first" << endl;
		return false;
	}

	if (!initialize()) return false;
	if (!computeVisibility()) return false;
	if (!extractComponents()) return false;
	if (StyleSimilarityConfig::mSegment_NParSDFMerging) {
		if (!mergeSDF()) return false;
	}

	return true;
}

bool SegmentSampleApxCvx::initialize() {

#ifdef OUTPUT_PROGRESS
	cout << "Building KD trees..." << endl;
#endif

	// build KD tree for ray intersection
	set<int> validFaces;
	for (int id : mpSamples->indices) validFaces.insert(id);
	mMeshTreeData.clear();
	for (int faceID = 0; faceID<(int)mpMesh->indices.size(); faceID++) {
		if (validFaces.find(faceID) == validFaces.end()) continue; // only add in valid faces
		vec3i idx = mpMesh->indices[faceID];
		G3D::Vector3 v0(mpMesh->positions[idx[0]].data());
		G3D::Vector3 v1(mpMesh->positions[idx[1]].data());
		G3D::Vector3 v2(mpMesh->positions[idx[2]].data());
		TKDTreeElement tri(TKDT::NamedTriangle(v0, v1, v2, faceID));
		mMeshTreeData.push_back(tri);
	}
	mMeshTree.init(mMeshTreeData.begin(), mMeshTreeData.end());

	// build KD tree for KNN

	if (!SampleUtil::buildKdTree(mpSamples->positions, mSampleTree, mSampleTreeData)) return false;

	// build kNN graph for samples

	if (!SegmentUtil::buildKNNGraph(*mpSamples, mSampleGraph, mSampleFlag)) return false;

	return true;
}

bool SegmentSampleApxCvx::computeVisibility() {

	int numPatches = (int)mPatches.size();
	mVisibility.resize(numPatches, vector<double>(numPatches, 0.0));

	// compute visibility for every pair of patches (only once)

#ifdef OUTPUT_PROGRESS
	cout << "Computing visibility..." << endl;
#endif

	float eps = mpSamples->radius * 0.01f;
	int pairCount = 0;
	int totalCount = numPatches*(numPatches - 1) / 2;

#pragma omp parallel for shared(pairCount)
	for(int srcID=0; srcID<numPatches-1; srcID++) {

		vector<int> srcSamples;
		if(!subsamplePatches(srcID, srcSamples)) {
			cout << "Error: sub-sampling patch " << srcID << endl;
			continue;
		}

		for(int dstID=srcID+1; dstID<numPatches; dstID++) {

#ifdef OUTPUT_PROGRESS
			if(pairCount%100==99) cout << "\rChecking pair " << (pairCount+1) << " / " << totalCount << "      ";
#pragma omp atomic
			pairCount++;
#endif

			vector<int> dstSamples;
			if(!subsamplePatches(dstID, dstSamples)) {
				cout << "Error: sub-sampling patch " << dstID << endl;
				continue;
			}

			int totalCount = 0;
			int visibleCount = 0;

			// check visibility for every pair of sample points
			for(int srcPointID: srcSamples) {
				vec3 srcP = mpSamples->positions[srcPointID];
				vec3 srcN = mpSamples->normals[srcPointID];
				for(int dstPointID : dstSamples) {
					vec3 dstP = mpSamples->positions[dstPointID];
					vec3 dstN = mpSamples->normals[dstPointID];

					totalCount++;
					vec3 rayDir = (dstP - srcP).normalize();
					float srcCos = cml::dot(srcN, rayDir);
					float dstCos = cml::dot(dstN, -rayDir);
					if( fabs(srcCos) < StyleSimilarityConfig::mSegment_NParCoplanarAngularThreshold &&
						fabs(dstCos) < StyleSimilarityConfig::mSegment_NParCoplanarAngularThreshold )
					{
						// co-planar, count as visible
						visibleCount++;
						continue;
					}
					if (srcCos > 0 || dstCos > 0) {
						// ray is outside the shape
						continue;
					}
					
					// check ray intersection
					
					float lineLength = (dstP - srcP).length();
					vec3 rayOrigin = srcP;
					bool rayFinished = false;

					while (!rayFinished) {
						rayOrigin = rayOrigin + rayDir * eps; // add eps for offset
						Thea::Ray3 ray(G3D::Vector3(rayOrigin.data()), G3D::Vector3(rayDir.data()));
						float dist = (float)mMeshTree.rayIntersectionTime(ray);
						if (dist < -0.5f) {
							// not hit? mark it as visible...
							rayFinished = true;
							break;
						}
						vec3 rayHitPos = rayOrigin + rayDir * dist;
						float rayLength = (rayHitPos - srcP).length();
						if (rayLength + eps > lineLength) {
							rayFinished = true;
							break;
						} else { // potentially hit
							double radius = mpSamples->radius * 1.5;
							SKDT::NamedPoint queryPoint(rayHitPos[0], rayHitPos[1], rayHitPos[2]);
							Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1);
							mSampleTree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult, radius);
							if (!queryResult.isEmpty()) {
								break;
							} // else: hit interior face, continue searching
						}

						rayOrigin = rayHitPos;
					}
					if (!rayFinished) {
						// ray is blocked by other faces
						continue;
					}

					visibleCount++;
				}
			}

			double visibility = visibleCount / (double)totalCount;
			mVisibility[srcID][dstID] = visibility;
			mVisibility[dstID][srcID] = visibility;
		}
	}

#ifdef OUTPUT_PROGRESS
	cout << endl;
#endif

	return true;
}

bool SegmentSampleApxCvx::extractComponents() {

#ifdef OUTPUT_PROGRESS
	cout << "Extracting weakly-convex components..." << endl;
#endif

	int numPatches = (int)mPatches.size();

	// initially assign each patch to one component
	mComponentIndices.resize(numPatches);
	for(int i=0; i<numPatches; i++) mComponentIndices[i] = i;
	mComponentPatches.resize(numPatches, vector<int>(1));
	for(int i=0; i<numPatches; i++) mComponentPatches[i][0] = i;

	// get a list of adjacent patches
	struct TFusion {
		int index1, index2;
		double visibility;
		TFusion(int i1, int i2, double v) : index1(i1), index2(i2), visibility(v) {}
	};
	vector<TFusion> fusionList;
	for(int srcID=0; srcID<numPatches; srcID++) {
		for(int dstID : mPatchNeighbors[srcID]) {
			if(srcID >= dstID) continue; // add only once
			fusionList.push_back(TFusion(srcID, dstID, mVisibility[srcID][dstID]));
		}
	}
	// sort by visibility ratio
	vector<int> fusionIndex(fusionList.size());
	for(int j=0; j<(int)fusionIndex.size(); j++) fusionIndex[j] = j;
	sort( fusionIndex.begin(), fusionIndex.end(), [&fusionList](int i1, int i2)
		{ return fusionList[i1].visibility > fusionList[i2].visibility; } ); // sort by visibility in descending order

	bool exportedFinestLevel = false;

	// iterate component fusion
	for(double theta : StyleSimilarityConfig::mSegment_NParVisibilityThresholdList.values) {

		// iterate segmentation

		for(int idx : fusionIndex) {
			TFusion &fusion = fusionList[idx];
			//if(fusion.visibility < theta) break; // no longer accepted
			int compID1 = mComponentIndices[fusion.index1];
			int compID2 = mComponentIndices[fusion.index2];
			if (compID1 == compID2 || compID1 < 0 || compID2 < 0) {
				// already in the same component or invalid patches
				continue;
			}
			// check visibility threshold
			double totalCount = 0;
			double visibleCount = 0;
			for(int patchID1 : mComponentPatches[compID1]) {
				for(int patchID2 : mComponentPatches[compID2]) {
					double visibility = mVisibility[patchID1][patchID2];
					double rays = (double)(mPatches[patchID1].size() * mPatches[patchID2].size());
					totalCount += rays;
					visibleCount += visibility * rays;
				}
			}
			if(visibleCount >= totalCount * theta) {
				// fuse components
				if( !mergeComponents(compID1, compID2) ) return false;
			}
		}

		if( !compactComponents() ) return false;

		// export segmentation result

		mSegmentsOffset.push_back((int)mSegments.size());
		if( !exportComponents(mSegments) ) return false;
		if (!exportedFinestLevel) {
			mFinestSegment = mSegments;
			if (!computeComponentGraph(mFinestSegmentGraph)) return false;
			exportedFinestLevel = true;
		}
	}

	return true;
}

bool SegmentSampleApxCvx::mergeSDF() {

#ifdef OUTPUT_PROGRESS
	cout << "Merging by SDF..." << endl;
#endif

	// UNDONE: param merging params following the ToG paper
	const double mergeCNCCVXRatio = 0.85;
	const double mergeDistRatio = 0.12;
	const double mergeFlatVisibility = 0.4;

	bool converged = false;
	int numIterations = 0;

	while (!converged) {

		converged = true;
		numIterations++;
		cout << "Iteration " << numIterations << endl;

		vector<vector<int>> segments(0);
		vector<vector<int>> segmentGraph;
		if (!exportComponents(segments)) return false;
		if (!computeComponentGraph(segmentGraph)) return false;
		int numSegments = (int)segments.size();

		// record ID of any patch associated with the component (for indexing segment during merging)
		vector<int> anyPatchInSegment(numSegments);
		for (int segID = 0; segID < numSegments; segID++) {
			anyPatchInSegment[segID] = mComponentPatches[segID][0];
		}

		Eigen::MatrixXd matSDFDistance;
		vector<bool> flatFlags;
		if (!computeSDFDistanceMatrix(segments, matSDFDistance, flatFlags)) return false;
		double maxSDFDistance = matSDFDistance.maxCoeff();

		Eigen::MatrixXi matCVXSeam, matCNCSeam;
		if (!computeSeamSet(segments, matCVXSeam, matCNCSeam)) return false;

		// merge non-flat components

		for (int segID1 = 0; segID1 < numSegments; segID1++) {
			if (flatFlags[segID1]) continue;
			for (int segID2 : segmentGraph[segID1]) {
				if (flatFlags[segID2]) continue;
				if (matCNCSeam(segID1, segID2) < mergeCNCCVXRatio * matCVXSeam(segID1, segID2) &&
					matSDFDistance(segID1, segID2) > 0 &&
					matSDFDistance(segID1, segID2) <= mergeDistRatio * maxSDFDistance)
				{
					int compID1 = mComponentIndices[anyPatchInSegment[segID1]];
					int compID2 = mComponentIndices[anyPatchInSegment[segID2]];
					if (!mergeComponents(compID1, compID2)) return false;
				}
			}
		}

		// merge flat components

		for (int segID1 = 0; segID1 < numSegments; segID1++) {
			// rule 1: max |CVX|, |CNC|=0, |CVX|>0
			int maxID = -1;
			double maxValue = 0;
			for (int segID2 : segmentGraph[segID1]) {
				if (segID1 >= matCNCSeam.rows() || segID2 >= matCNCSeam.cols()) {
					cout << "Error: " << segID1 << ", " << segID2 << ", ";
					cout << matCNCSeam.rows() << ", " << matCNCSeam.cols() << ", " << numSegments << endl;
				}
				if (matCNCSeam(segID1, segID2) == 0) {
					double nowValue = matCVXSeam(segID1, segID2);
					if (nowValue > maxValue) {
						maxValue = nowValue;
						maxID = segID2;
					}
				}
			}
			if (maxID < 0) {
				// rule 2: max |CVX|/|CNC|, |CVX|/|CNC|>=1, vis>=0.4
				for (int segID2 : segmentGraph[segID1]) {
					if (matCNCSeam(segID1, segID2) > 0) {
						double nowValue = matCVXSeam(segID1, segID2) / matCNCSeam(segID1, segID2);
						if (nowValue >= 1.0 && nowValue > maxValue) {

							// check visibility
							int compID1 = mComponentIndices[anyPatchInSegment[segID1]];
							int compID2 = mComponentIndices[anyPatchInSegment[segID2]];
							double totalCount = 0;
							double visibleCount = 0;
							for (int patchID1 : mComponentPatches[compID1]) {
								for (int patchID2 : mComponentPatches[compID2]) {
									double visibility = mVisibility[patchID1][patchID2];
									double rays = (double)(mPatches[patchID1].size() * mPatches[patchID2].size());
									totalCount += rays;
									visibleCount += visibility * rays;
								}
							}
							if (visibleCount >= mergeFlatVisibility * totalCount) {
								maxValue = nowValue;
								maxID = segID2;
							}

							// no visibility check
							//maxValue = nowValue;
							//maxID = segID2;
						}

					}
				}
			}

			if (maxID >= 0) {
				int compID1 = mComponentIndices[anyPatchInSegment[segID1]];
				int compID2 = mComponentIndices[anyPatchInSegment[maxID]];
				if (!mergeComponents(compID1, compID2)) return false;
				converged = false;
			}
		}

		if (!compactComponents()) return false;
	}

	// export segmentation result

	mSegmentsOffset.push_back((int)mSegments.size());
	if (!exportComponents(mSegments)) return false;
	
	return true;
}

bool SegmentSampleApxCvx::computeSDFDistanceMatrix(vector<vector<int>> &inSegments, Eigen::MatrixXd &outDistanceMatrix, vector<bool> &outFlatFlags) {

	int numSegments = (int)inSegments.size();

	const int sdfNumBins = 10;

	// compute SDF for patch

	vector<vector<double>> sdfDists(numSegments); // SDF value distribution
	outFlatFlags.assign(numSegments, false);
	double sdfMin = DBL_MAX;
	double sdfMax = -DBL_MAX;

	// global SDF

	vector<double> sampleSDFs;
	FeatureSDF fs(mpSamples, mpMesh, &sampleSDFs);
	if (!fs.calculate()) return false;
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		auto &segment = inSegments[segmentID];
		int numSamples = (int)segment.size();
		vector<double> &sdf = sdfDists[segmentID];
		sdf.assign(numSamples, 0);
		for (int k = 0; k < numSamples; k++) {
			sdf[k] = sampleSDFs[segment[k]];
			sdfMin = min(sdfMin, sdf[k]);
			sdfMax = max(sdfMax, sdf[k]);
		}
	}

	/*
	// local SDF

	const double sdfConeCosine = cos(cml::rad(60.0));
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		auto &segment = inSegments[segmentID];
		int numSamples = (int)segment.size();		
		int numSubSamples = min(numSamples, max(numSamples / 10, 100));
		vector<int> subSamples = segment;
		random_shuffle(subSamples.begin(), subSamples.end());

		vector<double> &sdf = sdfDists[segmentID];
		sdf.assign(numSubSamples, 0);
#pragma omp parallel for
		for (int subID = 0; subID < numSubSamples; subID++) {
			int id0 = subSamples[subID];
			vec3 p0 = mpSamples->positions[id0];
			vec3 n0 = mpSamples->normals[id0];
			vector<double> dists;
			for (int otherID = 0; otherID < numSubSamples; otherID++) {
				if (subID == otherID) continue;
				int id1 = subSamples[otherID];
				vec3 p1 = mpSamples->positions[id1];
				vec3 n1 = mpSamples->normals[id1];
				vec3 dir = p1 - p0;
				if (cml::dot(-n0, cml::normalize(dir)) > sdfConeCosine) {
					dists.push_back((double)(cml::dot(-n0, dir))); // ray length weighted by cosine
				}
			}
			if (dists.empty()) continue;
			int med = (int)dists.size() / 2;
			nth_element(dists.begin(), dists.begin() + med, dists.end()); // median
			sdf[subID] = dists[med];
		}
		bool valid = false;
		for (int subID = 0; subID < numSubSamples; subID++) {
			if (sdf[subID]) {
				sdfMin = min(sdfMin, sdf[subID]);
				sdfMax = max(sdfMax, sdf[subID]);
				valid = true;
			}
		}
		if (!valid) {
			outFlatFlags[segmentID] = true;
		}
	}
	*/

	if (sdfMin >= sdfMax) { // all flat
		outDistanceMatrix.setZero(numSegments, numSegments);
		return true;
	}

	// compute distance matrix

	vector<vector<double>> sdfHists(numSegments);
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		if (!FeatureUtil::computeHistogram(sdfDists[segmentID], sdfHists[segmentID], sdfNumBins, sdfMin, sdfMax)) return false;
	}

	outDistanceMatrix.setZero(numSegments, numSegments);
	for (int segID1 = 0; segID1 < numSegments - 1; segID1++) {
		if (outFlatFlags[segID1]) continue;
		for (int segID2 = segID1 + 1; segID2 < numSegments; segID2++) {
			if (outFlatFlags[segID2]) continue;
			double dist = FeatureUtil::computeEMD(sdfHists[segID1], sdfHists[segID2]);
			outDistanceMatrix(segID1, segID2) = dist;
			outDistanceMatrix(segID2, segID1) = dist;
		}
	}

	return true;
}

bool SegmentSampleApxCvx::computeSeamSet(vector<vector<int>> &inSegments, Eigen::MatrixXi &outCVXSeam, Eigen::MatrixXi &outCNCSeam) {

	int numSegments = (int)inSegments.size();

	vector<int> segmentMap(mpSamples->amount, -1);
	for (int segmentID = 0; segmentID < numSegments; segmentID++) {
		for (int sampleID : inSegments[segmentID]) {
			segmentMap[sampleID] = segmentID;
		}
	}

	outCVXSeam.resize(numSegments, numSegments);
	outCNCSeam.resize(numSegments, numSegments);
	outCVXSeam.setZero();
	outCNCSeam.setZero();
	for (int sampleID = 0; sampleID < mpSamples->amount; sampleID++) {
		int sampleSegmentID = segmentMap[sampleID];
		if (sampleSegmentID < 0) continue;

		vec3 sampleP = mpSamples->positions[sampleID];
		vec3 sampleN = mpSamples->normals[sampleID];

		for (int neighborID : mSampleGraph[sampleID]) {
			int neighborSegmentID = segmentMap[neighborID];
			if (neighborSegmentID < 0) continue;
			if (sampleSegmentID == neighborSegmentID) continue;

			vec3 neighborP = mpSamples->positions[neighborID];
			vec3 neighborN = mpSamples->normals[neighborID];

			float a1 = cml::dot(cml::normalize(sampleN), cml::normalize(neighborP - sampleP));
			float a2 = cml::dot(cml::normalize(neighborN), cml::normalize(sampleP - neighborP));
			const float eps = 0.01f;
			if (a1 <= eps && a2 <= eps) { // convex
				outCVXSeam(sampleSegmentID, neighborSegmentID)++;
				outCVXSeam(neighborSegmentID, sampleSegmentID)++;
			} else if (a1 > eps && a2 > eps) { // concave
				outCNCSeam(sampleSegmentID, neighborSegmentID)++;
				outCNCSeam(neighborSegmentID, sampleSegmentID)++;
			} // else: inconsistent neighbor normals
		}
	}

	return true;
}

bool SegmentSampleApxCvx::subsamplePatches(int patchID, vector<int> &outSamples) {

	int numVisSamples = StyleSimilarityConfig::mSegment_NParVisibilitySampleNumber;

	vector<int> index(mPatches[patchID].size());
	//for(int j=0; j<(int)index.size(); j++) index[j] = (j*521)%(int)index.size(); // definite sub-sampling for debug
	for(int j=0; j<(int)index.size(); j++) index[j] = j;
	random_shuffle(index.begin(), index.end());
	
	outSamples.resize(min((int)index.size(), numVisSamples));
	for(int j=0; j<(int)outSamples.size(); j++) {
		outSamples[j] = mPatches[patchID][index[j]];
	}

	return true;
}

bool SegmentSampleApxCvx::mergeComponents(int compID1, int compID2) {
	// merge compID1 and compID2

	if(compID1 == compID2) return true; // already merged
	if(compID1 > compID2) swap(compID1, compID2); // merge to lower ID

	for(int id : mComponentPatches[compID2]) mComponentIndices[id] = compID1;
	mComponentPatches[compID1].insert(mComponentPatches[compID1].end(), mComponentPatches[compID2].begin(), mComponentPatches[compID2].end());
	mComponentPatches[compID2].clear();

	return true;
}

bool SegmentSampleApxCvx::compactComponents() {

	// check compactness
	bool isCompact = true;
	for(auto &component : mComponentPatches) {
		if(component.empty()) {
			isCompact = false;
			break;
		}
	}
	if(isCompact) return true;

	// remove empty components
	vector<vector<int>> tmpComponents(mComponentPatches.begin(), mComponentPatches.end());
	vector<int> tmpMapping(tmpComponents.size(), -1);
	mComponentPatches.clear();
	for(int origCompID=0; origCompID<(int)tmpComponents.size(); origCompID++) {
		auto &component = tmpComponents[origCompID];
		if(!component.empty()) {
			tmpMapping[origCompID] = (int)mComponentPatches.size();
			mComponentPatches.push_back(component);
		}
	}

	// update patches' component indices
	for(int patchID=0; patchID<(int)mComponentIndices.size(); patchID++) {
		if (mComponentIndices[patchID] >= 0) {
			mComponentIndices[patchID] = tmpMapping[mComponentIndices[patchID]];
		}
	}

	return true;
}

bool SegmentSampleApxCvx::computeComponentGraph(vector<vector<int>> &outGraph) {

	int numComponents = (int)mComponentPatches.size();
	int numPatches = (int)mPatches.size();

	vector<int> patchMap(numPatches, -1); // component ID (-1 if not assigned) : # of patches
	for (int componentID = 0; componentID < numComponents; componentID++) {
		for (int patchID : mComponentPatches[componentID]) {
			patchMap[patchID] = componentID;
		}
	}

	vector<set<int>> nbSets(numComponents); // set of neighboring components : # of components
	for (auto &it : nbSets) it.clear();

	for (int patchID = 0; patchID < (int)mPatchNeighbors.size(); patchID++) {
		int patchCompID = patchMap[patchID];
		if (patchCompID < 0) continue;
		for (int neighborID : mPatchNeighbors[patchID]) {
			int neighborCompID = patchMap[neighborID];
			if (neighborCompID < 0) continue;
			if (patchCompID == neighborCompID) continue; // already in the same component
			nbSets[patchCompID].insert(neighborCompID);
			nbSets[neighborCompID].insert(patchCompID);
		}
	}

	outGraph.resize(numComponents);
	for (int componentID = 0; componentID < numComponents; componentID++) {
		auto &nbSet = nbSets[componentID];
		outGraph[componentID].assign(nbSet.begin(), nbSet.end());
	}

	return true;
}

bool SegmentSampleApxCvx::exportComponents(vector<vector<int>> &outComponents) {

	int minSampleNum = (int)( StyleSimilarityConfig::mSegment_NParPruningAreaRatio * mpSamples->amount / (int)mComponentPatches.size() );
	minSampleNum = max(minSampleNum, 20); // UNDONE: param minimum patch size

	for(auto &component : mComponentPatches) {
		if(component.empty()) continue;
		vector<int> samples(0);
		for(int patchID : component) {
			for(int sampleID : mPatches[patchID]) {
				samples.push_back(sampleID);
			}
		}
		if ((int)samples.size() < minSampleNum) { // too few sample points...
			component.clear();
			continue;
		}
		outComponents.push_back(samples);
	}

	if (!compactComponents()) return false;

	return true;
}

bool SegmentSampleApxCvx::exportSegmentation(vector<vector<int>> &outSegments) {

	auto itBegin = mSegments.begin();
	auto itEnd = mSegments.end();
	if (StyleSimilarityConfig::mSegment_NParOutputLastResult) {
		itBegin += mSegmentsOffset.back();
	}

	outSegments.reserve(outSegments.size() + (itEnd-itBegin));
	for (auto it = itBegin; it != itEnd; it++) {
		outSegments.push_back(*it);
	}

	return true;
}

bool SegmentSampleApxCvx::exportFinestSegmentation(vector<vector<int>> &outSegments, vector<vector<int>> &outGraph) {

	outSegments = mFinestSegment;
	outGraph = mFinestSegmentGraph;

	return true;
}

bool SegmentSampleApxCvx::visualizeSegmentation(string fileName) {

	// calculate visualization spacing

	vec3 visSpacing;
	{
		vec3 bbMin = mpSamples->positions[0];
		vec3 bbMax = bbMin;
		for(vec3 point : mpSamples->positions) {
			bbMin.minimize(point);
			bbMax.maximize(point);
		}

		visSpacing = (bbMax-bbMin) * 1.5f;
	}

	// export segmented shape

	PlyExporter pe;
	vec3 currentOffset;
	int segmentID = 0;
	currentOffset.zero();
	mSegmentsOffset.push_back((int)mSegments.size());
	for(int shapeID=1; shapeID<(int)mSegmentsOffset.size(); shapeID++) {
		
		auto &itBegin = shapeID ? mSegments.begin()+mSegmentsOffset[shapeID-1] : mPatches.begin();
		auto &itEnd = shapeID ? mSegments.begin()+mSegmentsOffset[shapeID] : mPatches.end();
		currentOffset = vec3(visSpacing[0]*(shapeID-2), 0, 0);
		for(auto &it=itBegin; it!=itEnd; it++) {

			vec3i color;
			if (shapeID == 0) {
				color = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
			} else {
				//color = vec3i((segmentID % 6) * 50, (segmentID / 6 % 6) * 50, (segmentID / 36) * 50); // color coding for debug
				color = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
				segmentID++;
			}

			vector<vec3> segmentPositions(0);
			vector<vec3> segmentNormals(0);
			for(int sampleID : (*it)) {
				segmentPositions.push_back( mpSamples->positions[sampleID] );
				segmentNormals.push_back( mpSamples->normals[sampleID] );
			}
			if( !pe.addPoint(&segmentPositions, &segmentNormals, currentOffset, color) ) return false;
		}
	}
	mSegmentsOffset.pop_back();

	if( !pe.output(fileName) ) return false;

	return true;
}
