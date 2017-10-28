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

#include "DemoIO.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>

#include "Mesh/MeshUtil.h"

#include "Sample/SamplePoissonDisk.h"
#include "Sample/SampleUtil.h"

#include "Segment/SegmentSampleSpectral.h"
#include "Segment/SegmentSampleApxCvx.h"
#include "Segment/SegmentUtil.h"

#include "Feature/FeatureAsset.h"
#include "Feature/FeatureUtil.h"

#include "Match/MatchSimpleICP.h"

#include "Element/ElementVoting.h"
#include "Element/ElementOptimization.h"
#include "Element/ElementDistance.h"
#include "Element/ElementMetric.h"
#include "Element/ElementUtil.h"

#include "Utility/PlyExporter.h"

#include "Data/StyleSimilarityData.h"
#include "Data/StyleSimilarityConfig.h"

using namespace std;
using namespace StyleSimilarity;

//#define REDO_EVERYTHING
#define OUTPUT_PROGRESS

#ifdef REDO_EVERYTHING
#define REDO true
#else
#define REDO false
#endif

bool DemoIO::process() {

	if (!runSinglePipeline()) return false;
	if (!runPairwisePipeline()) return false;
	if (!runTripletPipeline()) return false;

	return true;
}

bool DemoIO::runSinglePipeline() {

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	string meshListFileName = datasetPrefix + "mesh/mesh-list.txt";
	vector<string> meshNameList;
	ifstream meshListFile(meshListFileName);
	while (!meshListFile.eof()) {
		string line;
		getline(meshListFile, line);
		if (line.empty()) break;
		meshNameList.push_back(datasetPrefix + "mesh/" + StringUtil::trim(line));
	}
	meshListFile.close();

	int numMesh = (int)meshNameList.size();
	
	for (int meshID = 0; meshID < numMesh; meshID++) {

		string meshName = meshNameList[meshID];
		cout << "Pre-processing " << meshName << endl;
		if (!runSingleModelPreprocessing(meshName)) error("single model preprocessing error");
	}

	for (int meshID = 0; meshID < numMesh; meshID++) {

		string meshName = meshNameList[meshID];
		cout << "Segmentation " << meshName << endl;
		if (!runSingleModelSegmentation(meshName)) error("single model segmentation error");
	}

	for (int meshID = 0; meshID < numMesh; meshID++) {

		// don't parallelize this part:
		//     LFD uses OpenGL which requires rendering context for rendering
		//     rendering context can only be bound to the main thread in one process
		string meshName = meshNameList[meshID];
		cout << "Feature " << meshName << endl;
		if (!runSingleModelFeature(meshName)) error("single model feature error");
	}
	
	for (int meshID = 0; meshID < numMesh; meshID++) {

		string meshName = meshNameList[meshID];
		cout << "Saliency " << meshName << endl;
		if (!runSingleModelSaliency(meshID + 1, meshName)) error("single model saliency error");
	}
	
	return true;
}

bool DemoIO::runPairwisePipeline() {

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	string tripletFileName = datasetPrefix + "response/triplets.txt";

	vector<pair<string, string>> allPairs;
	ifstream tripletFile(tripletFileName);
	while (!tripletFile.eof()) {
		string line;
		getline(tripletFile, line);
		if (line.empty()) break;
		vector<string> tokens;
		StringUtil::split(line, ' ', tokens);
		if (tokens.size() != 3) {
			cout << "Error: invalid triplet " << line << endl;
			return false;
		}
		auto pair1 = make_pair(tokens[0], tokens[1]);
		auto pair2 = make_pair(tokens[0], tokens[2]);
		if (pair1.first > pair1.second) swap(pair1.first, pair1.second);
		if (pair2.first > pair2.second) swap(pair2.first, pair2.second);
		allPairs.push_back(pair1);
		allPairs.push_back(pair2);
	}
	tripletFile.close();

	vector<pair<string, string>> allElementPairs;
	for (int pairID = 0; pairID < (int)allPairs.size(); pairID++) {

		auto meshPair = allPairs[pairID];
		string meshName1 = datasetPrefix + "mesh/" + meshPair.first;
		string meshName2 = datasetPrefix + "mesh/" + meshPair.second;
		allElementPairs.push_back(make_pair(meshName1, meshName2));
		allElementPairs.push_back(make_pair(meshName2, meshName1));
	}

	for (int pairID = 0; pairID < (int)allElementPairs.size(); pairID++) {

		auto meshPair = allElementPairs[pairID];

		string name1 = meshPair.first;
		string name2 = meshPair.second;
		name1 = name1.substr(name1.find_last_of('/') + 1);
		name2 = name2.substr(name2.find_last_of('/') + 1);
		cout << "Voting " << name1 << " -- " << name2 << endl;

		if (!runPairedModelVoting(meshPair.first, meshPair.second)) error("paired model voting");
	}

	if (!StyleSimilarityConfig::mOptimization_FirstIteration) {
		// load weights...
		if (!ElementMetric::loadScaleSimplePatchDistance(datasetPrefix + "weight/data-scale-SPD.txt")) return false;
		if (!ElementMetric::loadScaleFullPatchDistance(datasetPrefix + "weight/data-scale-FPD.txt")) return false;
		if (!ElementMetric::loadScaleFullSaliency(datasetPrefix + "weight/data-scale-FS.txt")) return false;

		if (!ElementMetric::loadWeightsSimplePatchDistance(datasetPrefix + "weight/data-weight-SPD.txt")) return false;
		if (!ElementMetric::loadWeightsFullPatchDistance(datasetPrefix + "weight/data-weight-FPD.txt")) return false;
		if (!ElementMetric::loadWeightsFullSaliency(datasetPrefix + "weight/data-weight-FS.txt")) return false;

		// guarantee weight for point distance
		ElementMetric::mWeightsSimplePatchDistance(0) = ElementMetric::mWeightsSimplePatchDistance.maxCoeff();
	}

	for (int pairID = 0; pairID < (int)allElementPairs.size(); pairID++) {

		auto meshPair = allElementPairs[pairID];

		string name1 = meshPair.first;
		string name2 = meshPair.second;
		name1 = name1.substr(name1.find_last_of('/') + 1);
		name2 = name2.substr(name2.find_last_of('/') + 1);
		cout << "Processing " << name1 << " -- " << name2 << endl;

		if (!runPairedModelDistance(meshPair.first, meshPair.second)) error("paired model distance");
	}

	return true;
}

bool DemoIO::runTripletPipeline() {

	struct TTriplet {
		string meshA;
		string meshB;
		string meshC;
	};

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	vector<TTriplet> allTriplets;

	string tripletFileName = datasetPrefix + "response/triplets.txt";
	ifstream tripletFile(tripletFileName);
	while (!tripletFile.eof()) {
		string line;
		getline(tripletFile, line);
		if (line.empty()) break;
		vector<string> tokens;
		StringUtil::split(line, ' ', tokens);
		if (tokens.size() != 3) {
			cout << "Error: invalid triplet " << line << endl;
			return false;
		}

		TTriplet triplet;
		triplet.meshA = tokens[0];
		triplet.meshB = tokens[1];
		triplet.meshC = tokens[2];
		allTriplets.push_back(triplet);
	}
	tripletFile.close();

	int numTriplets = (int)allTriplets.size();
	for (int triID = 0; triID < numTriplets; triID++) {
		auto &triplet = allTriplets[triID];

		cout << "Processing " << triplet.meshA << " " << triplet.meshB << " " << triplet.meshC << endl;
		if (!runTripletDistance(triID+1, triplet.meshA, triplet.meshB, triplet.meshC)) return false;
	}


	return true;
}


bool DemoIO::runSingleModelPreprocessing(string meshName) {

	// names...

	string sampleName = meshName;
	sampleName = sampleName.replace(sampleName.find("mesh"), 4, "sample") + ".ply";
	if (!FileUtil::makedir(sampleName)) return false;

	meshName = meshName + ".ply";

	if (!REDO && FileUtil::existsfile(sampleName)) return true; // early quit

	// data...

	TTriangleMesh mesh;
	TSampleSet sample;

	if (!MeshUtil::loadMesh(meshName, mesh)) return false;
	if (!MeshUtil::recomputeNormals(mesh)) return false;
	
	// sampling

	if (REDO || !FileUtil::existsfile(sampleName)) {

#ifdef OUTPUT_PROGRESS
		cout << "Sampling..." << endl;
#endif

		int numSamples = StyleSimilarityConfig::mSample_WholeMeshSampleNumber;
		SamplePoissonDisk sp(&mesh);
		if (!sp.runSampling(numSamples)) return false;
		if (!sp.exportSample(sample)) return false;
		if (!SampleUtil::saveSample(sampleName, sample)) return false;
	}
	
	return true;
}

bool DemoIO::runSingleModelSegmentation(string meshName) {

	// names...

	string sampleName = meshName;
	sampleName = sampleName.replace(sampleName.find("mesh"), 4, "sample") + ".ply";
	if (!FileUtil::makedir(sampleName)) return false;

	string segmentPrefix = meshName;
	segmentPrefix.replace(segmentPrefix.find("mesh"), 4, "segment");
	if (!FileUtil::makedir(segmentPrefix)) return false;

	string patchName = segmentPrefix + "-patch.txt";
	string patchVName = segmentPrefix + "-patch.ply"; // for visualization
	string segmentName = segmentPrefix + "-segment.txt";
	string segmentVName = segmentPrefix + "-segment.ply"; // for visualization
	string finestSegmentName = segmentPrefix + "-finest-segment.txt";
	string finestSegmentVName = segmentPrefix + "-finest-segment.ply"; // for visualization
	string finestSegmentGName = segmentPrefix + "-finest-graph.ply"; // for visualization

	meshName = meshName + ".ply";

	if (!REDO && FileUtil::existsfile(finestSegmentName)) return true; // early quit

	// data...

	TTriangleMesh mesh;
	TSampleSet sample;
	vector<vector<int>> patch;
	vector<vector<int>> patchGraph;
	vector<vector<int>> segment;

	if (!MeshUtil::loadMesh(meshName, mesh)) return false;
	if (!MeshUtil::recomputeNormals(mesh)) return false;
	if (!SampleUtil::loadSample(sampleName, sample)) return false;

	// segmentation

	if (REDO || !FileUtil::existsfile(patchName)) {

#ifdef OUTPUT_PROGRESS
		cout << "Segmentation (initial patches)..." << endl;
#endif

		SegmentSampleSpectral sss(&sample, &mesh);
		if (!sss.runSegmentation()) return false;
		if (!sss.exportSegmentation(patch, patchGraph)) return false;
		if (!sss.visualizeSegmentation(patchVName)) return false;
		if (!SegmentUtil::savePatchData(patchName, patch, patchGraph)) return false;
	} else {
		if (!SegmentUtil::loadPatchData(patchName, patch, patchGraph)) return false;
		if (!FileUtil::existsfile(patchVName)) {
			if (!SegmentUtil::visualizeSegmentation(patchVName, sample, patch, true)) return false;
		}
	}

	if (REDO || !FileUtil::existsfile(segmentName)) {

#ifdef OUTPUT_PROGRESS
		cout << "Segmentation (multi-resolution)..." << endl;
#endif

		SegmentSampleApxCvx ssac(&sample, &mesh);
		if (!ssac.loadPatches(patch, patchGraph)) return false;
		if (!ssac.runSegmentation()) return false;
		if (!ssac.exportSegmentation(segment)) return false;
		if (!ssac.visualizeSegmentation(segmentVName)) return false;
		if (!SegmentUtil::saveSegmentationData(segmentName, segment)) return false;

		vector<vector<int>> finestSegment, finestSegmentGraph;
		if (!ssac.exportFinestSegmentation(finestSegment, finestSegmentGraph)) return false;
		if (!SegmentUtil::savePatchData(finestSegmentName, finestSegment, finestSegmentGraph)) return false;
		if (!SegmentUtil::visualizeSegmentation(finestSegmentVName, sample, finestSegment)) return false;
		if (!SegmentUtil::visualizeSegmentationGraph(finestSegmentGName, sample, finestSegment, finestSegmentGraph)) return false;

	} else {
		if (!SegmentUtil::loadSegmentationData(segmentName, segment)) return false;
		if (!FileUtil::existsfile(segmentVName)) {
			if (!SegmentUtil::visualizeSegmentation(segmentVName, sample, segment, true)) return false;
		}
	}

	if (REDO || !FileUtil::existsfile(finestSegmentName)) {

#ifdef OUTPUT_PROGRESS
		cout << "Segmentation (finest level)..." << endl;
#endif

		vector<vector<int>> finestSegment, finestSegmentGraph;
		if (!SegmentUtil::extractFinestSegmentation(sample, segment, finestSegment, finestSegmentGraph)) return false;
		if (!SegmentUtil::savePatchData(finestSegmentName, finestSegment, finestSegmentGraph)) return false;
		if (!SegmentUtil::visualizeSegmentation(finestSegmentVName, sample, finestSegment)) return false;
		if (!SegmentUtil::visualizeSegmentationGraph(finestSegmentGName, sample, finestSegment, finestSegmentGraph)) return false;
	}

	return true;
}

bool DemoIO::runSingleModelFeature(string meshName) {

	// names...

	string sampleName = meshName;
	sampleName = sampleName.replace(sampleName.find("mesh"), 4, "sample") + ".ply";

	string featurePrefix = meshName + "/";
	featurePrefix.replace(featurePrefix.find("mesh"), 4, "feature");
	if (!FileUtil::makedir(featurePrefix)) return false;

	string featureCurvatureName = featurePrefix + "curvature.txt";
	string featureCurvatureVName = featurePrefix + "curvature.ply"; // for visualization
	string featureSDFName = featurePrefix + "SDF.txt";
	string featureSDFVName = featurePrefix + "SDF.ply"; // for visualization
	string featureAOName = featurePrefix + "AO.txt";
	string featureAOVName = featurePrefix + "AO.ply"; // for visualization
	string featureSDName = featurePrefix + "SD.txt";
	string featureSDVName = featurePrefix + "SD.ply"; // for visualization
	string featureCurvePrefix = featurePrefix + "curve";
	string featureTalFPFHName = featurePrefix + "Tal-FPFH.txt";
	string featureTalFPFHVName = featurePrefix + "Tal-FPFH.ply"; // for visualization
	string featureTalSIName = featurePrefix + "Tal-SI.txt";
	string featureTalSIVName = featurePrefix + "Tal-SI.ply"; // for visualization
	string featureTalSCName = featurePrefix + "Tal-SC.txt";
	string featureTalSCVName = featurePrefix + "Tal-SC.ply"; // for visualization
	string featureLFDName = featurePrefix + "LFD.txt";
	string featureLFDVName = featurePrefix + "LFD.ppm"; // for visualization
	string featureGeodesicName = featurePrefix + "geodesic.txt";
	string featureGeodesicVName = featurePrefix + "geodesic.ply"; // for visualization

	meshName = meshName + ".ply";

	if (!REDO && FileUtil::existsfile(featureGeodesicName)) return true; // early quit

	// data...

	TTriangleMesh mesh;
	TSampleSet sample;

	if (!MeshUtil::loadMesh(meshName, mesh)) return false;
	if (!MeshUtil::recomputeNormals(mesh)) return false;
	if (!SampleUtil::loadSample(sampleName, sample)) return false;

	// feature

	if (REDO || !FileUtil::existsfile(featureCurvatureName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: curvature..." << endl;
#endif
		vector<FeatureCurvature::TCurvature> feature;
		FeatureCurvature fc(&sample, &feature);
		if (!fc.calculate()) return false;
		if (!fc.saveFeature(featureCurvatureName)) return false;
		if (!fc.visualize(featureCurvatureVName)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureSDFName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Shape Diameter Feature..." << endl;
#endif
		vector<double> feature;
		FeatureSDF fs(&sample, &mesh, &feature);
		if (!fs.calculate()) return false;
		if (!fs.visualize(featureSDFVName)) return false;
		if (!FeatureAsset::saveFeature(featureSDFName, feature)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureAOName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Ambient Occlusion..." << endl;
#endif
		vector<double> feature;
		FeatureAmbientOcclusion fa(&sample, &mesh, &feature);
		if (!fa.calculate()) return false;
		if (!fa.visualize(featureAOVName)) return false;
		if (!FeatureAsset::saveFeature(featureAOName, feature)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureSDName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Shape Distributions..." << endl;
#endif
		vector<double> feature;
		FeatureShapeDistributions fs(&sample, &feature);
		if (!fs.calculate()) return false;
		if (!fs.visualize(featureSDVName)) return false;
		if (!FeatureAsset::saveFeature(featureSDName, feature)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureCurvePrefix + "-0.ply")) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: curve..." << endl;
#endif
		vector<TPointSet> curves;
		CurveRidgeValley rv;
		rv.loadMesh(&mesh);
		if (!rv.extractCurve()) return false;
		if (!rv.exportPoints(curves, sample.radius*0.5)) return false;
		int curveID = 0;
		for (auto &curve : curves) {
			PlyExporter pe;
			if (!pe.addPoint(&curve.positions, &curve.normals)) return false;
			if (!pe.output(featureCurvePrefix + "-" + to_string(curveID) + ".ply")) return false;
			curveID++;
		}
	}

	if (REDO || !FileUtil::existsfile(featureTalFPFHName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Tal FPFH..." << endl;
#endif
		vector<vector<double>> feature;
		FeatureTalSaliency ft(&sample, &feature);
		if (!ft.calculate<FeatureFPFH>()) return false;
		if (!ft.saveFeature(featureTalFPFHName)) return false;
		if (!ft.visualize(featureTalFPFHVName)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureTalSIName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Tal Spin Images..." << endl;
#endif
		vector<vector<double>> feature;
		FeatureTalSaliency ft(&sample, &feature);
		if (!ft.calculate<FeatureSpinImages>()) return false;
		if (!ft.saveFeature(featureTalSIName)) return false;
		if (!ft.visualize(featureTalSIVName)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureTalSCName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Tal Shape Contexts..." << endl;
#endif
		vector<vector<double>> feature;
		FeatureTalSaliency ft(&sample, &feature);
		if (!ft.calculate<FeatureShapeContexts>()) return false;
		if (!ft.saveFeature(featureTalSCName)) return false;
		if (!ft.visualize(featureTalSCVName)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureLFDName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: Light Field Descriptor..." << endl;
#endif
		vector<double> feature;
		FeatureLFD fl(&mesh, &feature);
		if (!fl.calculate()) return false;
		if (!fl.visualize(featureLFDVName)) return false;
		if (!FeatureAsset::saveFeature(featureLFDName, feature)) return false;
	}

	if (REDO || !FileUtil::existsfile(featureGeodesicName)) {
#ifdef OUTPUT_PROGRESS
		cout << "Feature: geodesic..." << endl;
#endif
		vector<double> feature;
		FeatureGeodesic fg(&sample, &feature);
		if (!fg.calculate()) return false;
		if (!fg.visualize(featureGeodesicVName)) return false;
		if (!FeatureAsset::saveFeature(featureGeodesicName, feature)) return false;
	}

	return true;
}

bool DemoIO::runSingleModelSaliency(int shapeID, string meshName) {

	// names...

	string sampleName = meshName;
	sampleName = sampleName.replace(sampleName.find("mesh"), 4, "sample") + ".ply";

	string featurePrefix = meshName + "/";
	featurePrefix.replace(featurePrefix.find("mesh"), 4, "feature");

	string backupSaliencyFile = featurePrefix + "saliency.txt";
	string saliencyFile = StyleSimilarityConfig::mData_DataSetRootFolder + "saliency/" + to_string(shapeID) + ".txt";

	meshName = meshName + ".ply";

	if (!REDO && FileUtil::existsfile(saliencyFile)) return true; // early quit
	if (!FileUtil::makedir(saliencyFile)) return false;

	// data...

	TTriangleMesh mesh;
	TSampleSet sample;
	FeatureAsset feature;

	// mesh
	if (!MeshUtil::loadMesh(meshName, mesh)) return false;

	// sample
	if (!SampleUtil::loadSample(sampleName, sample)) return false;

	// feature
	if (!feature.loadCurvature(featurePrefix + "curvature.txt")) return false;
	if (!feature.loadSDF(featurePrefix + "SDF.txt")) return false;
	if (!feature.loadAO(featurePrefix + "AO.txt")) return false;
	if (!feature.loadSD(featurePrefix + "SD.txt")) return false;
	if (!feature.loadCurve(featurePrefix + "curve")) return false;
	if (!feature.loadTalFPFH(featurePrefix + "Tal-FPFH.txt")) return false;
	if (!feature.loadTalSI(featurePrefix + "Tal-SI.txt")) return false;
	if (!feature.loadTalSC(featurePrefix + "Tal-SC.txt")) return false;
	if (!feature.loadLFD(featurePrefix + "LFD.txt")) return false;
	if (!feature.loadGeodesic(featurePrefix + "geodesic.txt")) return false;

	Eigen::MatrixXd outSaliency;
	if (!ElementMetric::computePointSaliency(mesh, sample, feature, outSaliency)) return false;
	if (!ElementUtil::saveMatrixBinary(saliencyFile, outSaliency)) return false;

	if (!FileUtil::existsfile(backupSaliencyFile)) {
		if (!FileUtil::makedir(backupSaliencyFile)) return false;
		if (!FileUtil::copyfile(saliencyFile, backupSaliencyFile)) return false;
	}

	return true;
}

bool DemoIO::runPairedModelVoting(string sourceMeshName, string targetMeshName) {

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	// names...

	string sourceSampleName = sourceMeshName;
	string targetSampleName = targetMeshName;
	sourceSampleName = sourceSampleName.replace(sourceSampleName.find("mesh"), 4, "sample") + ".ply";
	targetSampleName = targetSampleName.replace(targetSampleName.find("mesh"), 4, "sample") + ".ply";

	string sourceSegmentPrefix = sourceMeshName;
	string targetSegmentPrefix = targetMeshName;
	sourceSegmentPrefix.replace(sourceSegmentPrefix.find("mesh"), 4, "segment");
	targetSegmentPrefix.replace(targetSegmentPrefix.find("mesh"), 4, "segment");
	string sourcePatchName = sourceSegmentPrefix + "-finest-segment.txt";
	string targetPatchName = targetSegmentPrefix + "-finest-segment.txt";
	string sourceSegmentName = sourceSegmentPrefix + "-segment.txt";
	string targetSegmentName = targetSegmentPrefix + "-segment.txt";

	string sourceFeaturePrefix = sourceMeshName + "/";
	string targetFeaturePrefix = targetMeshName + "/";
	sourceFeaturePrefix.replace(sourceFeaturePrefix.find("mesh"), 4, "feature");
	targetFeaturePrefix.replace(targetFeaturePrefix.find("mesh"), 4, "feature");

	string elementPrefix = datasetPrefix + "element/";
	elementPrefix += sourceMeshName.substr(sourceMeshName.find_last_of("/\\") + 1) + "--";
	elementPrefix += targetMeshName.substr(targetMeshName.find_last_of("/\\") + 1) + "/";
	if (!FileUtil::makedir(elementPrefix)) return false;

	string dataVoteName = elementPrefix + "data-votes.txt";

	sourceMeshName = sourceMeshName + ".ply";
	targetMeshName = targetMeshName + ".ply";

	if (!REDO && FileUtil::existsfile(dataVoteName)) return true; // early quit

	// data...

	StyleSimilarityData data;
	
#ifdef OUTPUT_PROGRESS
	cout << "Element: prepairing data..." << endl;
#endif

	if (!MeshUtil::loadMesh(sourceMeshName, data.mSourceMesh)) return false;
	if (!MeshUtil::loadMesh(targetMeshName, data.mTargetMesh)) return false;

	if (!SampleUtil::loadSample(sourceSampleName, data.mSourceSamples)) return false;
	if (!SampleUtil::loadSample(targetSampleName, data.mTargetSamples)) return false;

	if (!SegmentUtil::loadPatchData(sourcePatchName, data.mSourcePatchesIndices, data.mSourcePatchesGraph)) return false;
	if (!SegmentUtil::loadPatchData(targetPatchName, data.mTargetPatchesIndices, data.mTargetPatchesGraph)) return false;

	if (!SegmentUtil::extractSampleSet(data.mSourceSamples, data.mSourcePatchesIndices, data.mSourcePatches)) return false;
	if (!SegmentUtil::extractSampleSet(data.mTargetSamples, data.mTargetPatchesIndices, data.mTargetPatches)) return false;
	
	if (!SegmentUtil::loadSegmentationData(sourceSegmentName, data.mSourceSegmentsIndices)) return false;
	if (!SegmentUtil::loadSegmentationData(targetSegmentName, data.mTargetSegmentsIndices)) return false;

	if (!SegmentUtil::extractSampleSet(data.mSourceSamples, data.mSourceSegmentsIndices, data.mSourceSegments)) return false;
	if (!SegmentUtil::extractSampleSet(data.mTargetSamples, data.mTargetSegmentsIndices, data.mTargetSegments)) return false;

	// element

	if (!SampleUtil::buildKdTree(data.mSourceSamples.positions, data.mSourceSamplesKdTree, data.mSourceSamplesKdTreeData)) return false;
	if (!SampleUtil::buildKdTree(data.mTargetSamples.positions, data.mTargetSamplesKdTree, data.mTargetSamplesKdTreeData)) return false;
	
	ElementVoting av(&data);

	if (REDO || !FileUtil::existsfile(dataVoteName)) {
		if (!av.computeVotes())  return false;
		if (!av.saveVotes(dataVoteName)) return false;
	}

	return true;
}

bool DemoIO::runPairedModelDistance(string sourceMeshName, string targetMeshName) {

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	// names...

	string sourceSampleName = sourceMeshName;
	string targetSampleName = targetMeshName;
	sourceSampleName = sourceSampleName.replace(sourceSampleName.find("mesh"), 4, "sample") + ".ply";
	targetSampleName = targetSampleName.replace(targetSampleName.find("mesh"), 4, "sample") + ".ply";

	string sourceSegmentPrefix = sourceMeshName;
	string targetSegmentPrefix = targetMeshName;
	sourceSegmentPrefix.replace(sourceSegmentPrefix.find("mesh"), 4, "segment");
	targetSegmentPrefix.replace(targetSegmentPrefix.find("mesh"), 4, "segment");
	string sourcePatchName = sourceSegmentPrefix + "-finest-segment.txt";
	string targetPatchName = targetSegmentPrefix + "-finest-segment.txt";
	string sourceSegmentName = sourceSegmentPrefix + "-segment.txt";
	string targetSegmentName = targetSegmentPrefix + "-segment.txt";

	string sourceFeaturePrefix = sourceMeshName + "/";
	string targetFeaturePrefix = targetMeshName + "/";
	sourceFeaturePrefix.replace(sourceFeaturePrefix.find("mesh"), 4, "feature");
	targetFeaturePrefix.replace(targetFeaturePrefix.find("mesh"), 4, "feature");

	string elementPrefix = datasetPrefix + "element/";
	elementPrefix += sourceMeshName.substr(sourceMeshName.find_last_of("/\\") + 1) + "--";
	elementPrefix += targetMeshName.substr(targetMeshName.find_last_of("/\\") + 1) + "/";
	if (!FileUtil::makedir(elementPrefix)) return false;

	string dataVoteName = elementPrefix + "data-votes.txt";
	string dataVoteDistanceName = elementPrefix + "data-vote-distance.txt";
	string dataGlobalAlignmentName = elementPrefix + "data-global-alignment.txt";

	sourceMeshName = sourceMeshName + ".ply";
	targetMeshName = targetMeshName + ".ply";

	if (true) {
		// early quit
		stringstream ss;
		ss.precision(1);
		ss << "-" << StyleSimilarityConfig::mOptimization_DistanceSigmaList.values.back();
		if (!REDO && FileUtil::existsfile(elementPrefix + "data-index-ut" + ss.str() + ".txt")) return true;
	}

	// data...

	StyleSimilarityData data;

#ifdef OUTPUT_PROGRESS
	cout << "Element: prepairing data..." << endl;
#endif

	if (!MeshUtil::loadMesh(sourceMeshName, data.mSourceMesh)) return false;
	if (!MeshUtil::loadMesh(targetMeshName, data.mTargetMesh)) return false;

	if (!SampleUtil::loadSample(sourceSampleName, data.mSourceSamples)) return false;
	if (!SampleUtil::loadSample(targetSampleName, data.mTargetSamples)) return false;

	if (!SegmentUtil::loadPatchData(sourcePatchName, data.mSourcePatchesIndices, data.mSourcePatchesGraph)) return false;
	if (!SegmentUtil::loadPatchData(targetPatchName, data.mTargetPatchesIndices, data.mTargetPatchesGraph)) return false;

	if (!SegmentUtil::extractSampleSet(data.mSourceSamples, data.mSourcePatchesIndices, data.mSourcePatches)) return false;
	if (!SegmentUtil::extractSampleSet(data.mTargetSamples, data.mTargetPatchesIndices, data.mTargetPatches)) return false;

	if (!SegmentUtil::loadSegmentationData(sourceSegmentName, data.mSourceSegmentsIndices)) return false;
	if (!SegmentUtil::loadSegmentationData(targetSegmentName, data.mTargetSegmentsIndices)) return false;

	if (!SegmentUtil::extractSampleSet(data.mSourceSamples, data.mSourceSegmentsIndices, data.mSourceSegments)) return false;
	if (!SegmentUtil::extractSampleSet(data.mTargetSamples, data.mTargetSegmentsIndices, data.mTargetSegments)) return false;

	if (!SampleUtil::buildKdTree(data.mSourceSamples.positions, data.mSourceSamplesKdTree, data.mSourceSamplesKdTreeData)) return false;
	if (!SampleUtil::buildKdTree(data.mTargetSamples.positions, data.mTargetSamplesKdTree, data.mTargetSamplesKdTreeData)) return false;

	data.mpSourceFeatures = new FeatureAsset();
	data.mpTargetFeatures = new FeatureAsset();

	FeatureAsset &srcFeatures = *(data.mpSourceFeatures);
	if (!srcFeatures.loadCurvature(sourceFeaturePrefix + "curvature.txt")) return false;
	if (!srcFeatures.loadSDF(sourceFeaturePrefix + "SDF.txt")) return false;
	if (!srcFeatures.loadAO(sourceFeaturePrefix + "AO.txt")) return false;
	if (!srcFeatures.loadSD(sourceFeaturePrefix + "SD.txt")) return false;
	if (!srcFeatures.loadCurve(sourceFeaturePrefix + "curve")) return false;
	if (!srcFeatures.loadTalFPFH(sourceFeaturePrefix + "Tal-FPFH.txt")) return false;
	if (!srcFeatures.loadTalSI(sourceFeaturePrefix + "Tal-SI.txt")) return false;
	if (!srcFeatures.loadTalSC(sourceFeaturePrefix + "Tal-SC.txt")) return false;
	if (!srcFeatures.loadLFD(sourceFeaturePrefix + "LFD.txt")) return false;
	if (!srcFeatures.loadGeodesic(sourceFeaturePrefix + "geodesic.txt")) return false;
	if (!srcFeatures.loadSaliency(sourceFeaturePrefix + "saliency.txt")) return false;

	FeatureAsset &tgtFeatures = *(data.mpTargetFeatures);
	if (!tgtFeatures.loadCurvature(targetFeaturePrefix + "curvature.txt")) return false;
	if (!tgtFeatures.loadSDF(targetFeaturePrefix + "SDF.txt")) return false;
	if (!tgtFeatures.loadAO(targetFeaturePrefix + "AO.txt")) return false;
	if (!tgtFeatures.loadSD(targetFeaturePrefix + "SD.txt")) return false;
	if (!tgtFeatures.loadCurve(targetFeaturePrefix + "curve")) return false;
	if (!tgtFeatures.loadTalFPFH(targetFeaturePrefix + "Tal-FPFH.txt")) return false;
	if (!tgtFeatures.loadTalSI(targetFeaturePrefix + "Tal-SI.txt")) return false;
	if (!tgtFeatures.loadTalSC(targetFeaturePrefix + "Tal-SC.txt")) return false;
	if (!tgtFeatures.loadLFD(targetFeaturePrefix + "LFD.txt")) return false;
	if (!tgtFeatures.loadGeodesic(targetFeaturePrefix + "geodesic.txt")) return false;
	if (!tgtFeatures.loadSaliency(targetFeaturePrefix + "saliency.txt")) return false;

	// distance...

	ElementVoting av(&data);
	
	if (!FileUtil::existsfile(dataVoteName)) {
		error("Run voting first");
	} else {
		if (!av.loadVotes(dataVoteName)) return false;
	}

	if (REDO || !FileUtil::existsfile(dataVoteDistanceName)) {
		if (!av.computeVoteDistances())  return false;
		if (!av.saveVoteDistance(dataVoteDistanceName)) return false;
	} else {
		if (!av.loadVoteDistance(dataVoteDistanceName)) return false;
	}

	if (REDO || !FileUtil::existsfile(dataGlobalAlignmentName)) {
		if (!av.computeGlobalAlignment())  return false;
		if (!av.saveGlobalAlignment(dataGlobalAlignmentName)) return false;
	} else {
		if (!av.loadGlobalAlignment(dataGlobalAlignmentName)) return false;
	}
	
	if (StyleSimilarityConfig::mOptimization_FirstIteration) { // init weights for first iteration

		string dataWeightSPDName;
		string dataScaleSPDName;

		Eigen::MatrixXd matVD;
		ElementUtil::loadMatrixBinary(dataVoteDistanceName, matVD);
		int n = (int)matVD.rows();
		int dim = (int)matVD.cols();
		if (n > 0 && dim > 0) {

			dataWeightSPDName = elementPrefix + "data-weight-SPD.txt";
			dataScaleSPDName = elementPrefix + "data-scale-SPD.txt";

			vector<double> vecW(dim);
			vector<double> vecS(dim);
			for (int d = 0; d < dim; d++) {
				vector<double> vecD(n);
				for (int k = 0; k < n; k++) vecD[k] = matVD(k, d);
				int m = (int)(n*StyleSimilarityConfig::mOptimization_DistanceSigmaPercentile);
				nth_element(vecD.begin(), vecD.begin() + m, vecD.end());
				vecW[d] = 1.0;
				vecS[d] = vecD[m];
			}
			for (int d = 2; d < dim; d++) vecW[d] = 0; // only keeps weights for ICP position & normal distance
			ofstream wfile(dataWeightSPDName);
			for (int d = 0; d < dim; d++) {
				wfile << vecW[d] << endl;
			}
			wfile.close();
			ofstream sfile(dataScaleSPDName);
			for (int d = 0; d < dim; d++) {
				sfile << vecS[d] << endl;
			}
			sfile.close();

			// copy any weight file to weight folder in case of 0 vote for other pairs

			string globalWeightSPDName = datasetPrefix + "weight/data-weight-SPD.txt";
			string globalScaleSPDName = datasetPrefix + "weight/data-scale-SPD.txt";
			if (!FileUtil::existsfile(globalWeightSPDName)) {
				if (!FileUtil::makedir(globalWeightSPDName)) return false;
				if (!FileUtil::copyfile(dataWeightSPDName, globalWeightSPDName)) return false;
			}
			if (!FileUtil::existsfile(globalScaleSPDName)) {
				if (!FileUtil::makedir(globalScaleSPDName)) return false;
				if (!FileUtil::copyfile(dataScaleSPDName, globalScaleSPDName)) return false;
			}

		} else {
			dataWeightSPDName = datasetPrefix + "weight/data-weight-SPD.txt";
			dataScaleSPDName = datasetPrefix + "weight/data-scale-SPD.txt";
		}

		if (!ElementMetric::loadWeightsSimplePatchDistance(dataWeightSPDName)) return false;
		if (!ElementMetric::loadScaleSimplePatchDistance(dataScaleSPDName)) return false;
	}
	

	for (double sigma : StyleSimilarityConfig::mOptimization_DistanceSigmaList.values) {

		stringstream ss;
		ss.precision(1);
		ss << "-" << sigma;
		string affix = ss.str();

		string dataModeName = elementPrefix + "data-modes" + affix + ".txt";
		string dataOldModeName = elementPrefix + "data-modes-old" + affix + ".txt";
		string dataElementName = elementPrefix + "data-element" + affix + ".txt";
		string anyFinalName = elementPrefix + "data-index-ut" + affix + ".txt";

		if (REDO || !FileUtil::existsfile(dataModeName)) {
			if (!av.clusterVotes(sigma))  return false;
			if (!av.adjustModes())  return false;
			if (!av.saveModes(dataModeName)) return false;
		} else {
			if (!av.loadModes(dataModeName)) return false;
		}

		if (REDO || !FileUtil::existsfile(dataElementName)) {
			ElementOptimization ao(&data);
			if (!av.saveModes(dataOldModeName)) return false;
			if (!ao.process()) return false;
			if (!ao.output(dataElementName)) return false;
			if (!ao.visualize(elementPrefix, affix)) return false;
			if (!av.saveModes(dataModeName)) return false; // update adjusted transformation by element result
		}

		// distance

		if (REDO || !FileUtil::existsfile(anyFinalName)) {

			ElementDistance ad(&data);
			if (!ad.loadElement(dataElementName)) return false;
			if (!ad.process(elementPrefix, affix)) return false;
		}
	}

	if (data.mpSourceFeatures) {
		delete data.mpSourceFeatures;
		data.mpSourceFeatures = 0;
	}
	if (data.mpTargetFeatures) {
		delete data.mpTargetFeatures;
		data.mpTargetFeatures = 0;
	}

	return true;
}

bool DemoIO::runTripletDistance(int tripletID, string meshA, string meshB, string meshC) {

	string datasetPrefix = StyleSimilarityConfig::mData_DataSetRootFolder;

	// names...

	string shortNameA = meshA.substr(meshA.find_last_of('/') + 1);
	string shortNameB = meshB.substr(meshB.find_last_of('/') + 1);
	string shortNameC = meshC.substr(meshC.find_last_of('/') + 1);
	
	vector<string> pairNames;
	pairNames.push_back(shortNameA + "--" + shortNameB);
	pairNames.push_back(shortNameB + "--" + shortNameA);
	pairNames.push_back(shortNameA + "--" + shortNameC);
	pairNames.push_back(shortNameC + "--" + shortNameA);

	string folderPrefix = datasetPrefix + "triplet/" + to_string(tripletID) + "/";
	if (!FileUtil::makedir(folderPrefix)) return false;

	// process files

	for (double sigma : StyleSimilarityConfig::mOptimization_DistanceSigmaList.values) {

		stringstream ss;
		ss.precision(1);
		ss << "-" << sigma;
		string affix = ss.str();

		if (!REDO && FileUtil::existsfile(folderPrefix + "utAC" + affix + ".txt")) continue; // early quit

		// load data

		Eigen::MatrixXd metricPD[4];
		Eigen::RowVectorXd metricSD[4];
		vector<vector<int>> indexAS[4], indexAT[4];
		vector<int> indexUS[4], indexUT[4];

		for (int pairID = 0; pairID < 4; pairID++) {
			string filePrefix = datasetPrefix + "element/" + pairNames[pairID] + "/";

			if (!ElementUtil::loadMatrixBinary(filePrefix + "data-metric-pd" + affix + ".txt", metricPD[pairID])) return false;
			if (!ElementUtil::loadRowVectorBinary(filePrefix + "data-metric-sd" + affix + ".txt", metricSD[pairID])) return false;
			if (!ElementUtil::loadCellArraysBinary(filePrefix + "data-index-as" + affix + ".txt", indexAS[pairID])) return false;
			if (!ElementUtil::loadCellArraysBinary(filePrefix + "data-index-at" + affix + ".txt", indexAT[pairID])) return false;
			if (!ElementUtil::loadCellArrayBinary(filePrefix + "data-index-us" + affix + ".txt", indexUS[pairID])) return false;
			if (!ElementUtil::loadCellArrayBinary(filePrefix + "data-index-ut" + affix + ".txt", indexUT[pairID])) return false;

			sort(indexUS[pairID].begin(), indexUS[pairID].end());
			sort(indexUT[pairID].begin(), indexUT[pairID].end());
		}

		// combine data

		Eigen::MatrixXd pdAB(metricPD[0].rows() + metricPD[1].rows(), max(metricPD[0].cols(), metricPD[1].cols()));
		if (metricPD[0].rows() && metricPD[1].rows()) {
			pdAB << metricPD[0], metricPD[1];
		} else if (metricPD[0].rows()) {
			pdAB << metricPD[0];
		} else {
			pdAB << metricPD[1];
		}
		Eigen::MatrixXd pdAC(metricPD[2].rows() + metricPD[3].rows(), max(metricPD[2].cols(), metricPD[3].cols()));
		if (metricPD[2].rows() && metricPD[3].rows()) {
			pdAC << metricPD[2], metricPD[3];
		} else if (metricPD[2].rows()) {
			pdAC << metricPD[2];
		} else {
			pdAC << metricPD[3];
		}

		Eigen::RowVectorXd sdAB = (metricSD[0] + metricSD[1]) / 2;
		Eigen::RowVectorXd sdAC = (metricSD[2] + metricSD[3]) / 2;

		vector<vector<int>> asAB = indexAS[0];
		asAB.insert(asAB.end(), indexAT[1].begin(), indexAT[1].end());
		vector<vector<int>> asAC = indexAS[2];
		asAC.insert(asAC.end(), indexAT[3].begin(), indexAT[3].end());

		vector<vector<int>> atAB = indexAT[0];
		atAB.insert(atAB.end(), indexAS[1].begin(), indexAS[1].end());
		vector<vector<int>> atAC = indexAT[2];
		atAC.insert(atAC.end(), indexAS[3].begin(), indexAS[3].end());

		vector<int>::iterator it;

		vector<int> usAB(indexUS[0].size() + indexUT[1].size());
		it = set_intersection(indexUS[0].begin(), indexUS[0].end(), indexUT[1].begin(), indexUT[1].end(), usAB.begin());
		usAB.resize(it - usAB.begin());
		vector<int> usAC(indexUS[2].size() + indexUT[3].size());
		it = set_intersection(indexUS[2].begin(), indexUS[2].end(), indexUT[3].begin(), indexUT[3].end(), usAC.begin());
		usAC.resize(it - usAC.begin());

		vector<int> utAB(indexUT[0].size() + indexUS[1].size());
		it = set_intersection(indexUT[0].begin(), indexUT[0].end(), indexUS[1].begin(), indexUS[1].end(), utAB.begin());
		utAB.resize(it - utAB.begin());
		vector<int> utAC(indexUT[2].size() + indexUS[3].size());
		it = set_intersection(indexUT[2].begin(), indexUT[2].end(), indexUS[3].begin(), indexUS[3].end(), utAC.begin());
		utAC.resize(it - utAC.begin());

		// output data

		if (!ElementUtil::saveMatrixBinary(folderPrefix + "pdAB" + affix + ".txt", pdAB)) return false;
		if (!ElementUtil::saveMatrixBinary(folderPrefix + "pdAC" + affix + ".txt", pdAC)) return false;
		if (!ElementUtil::saveRowVectorBinary(folderPrefix + "sdAB" + affix + ".txt", sdAB)) return false;
		if (!ElementUtil::saveRowVectorBinary(folderPrefix + "sdAC" + affix + ".txt", sdAC)) return false;
		if (!ElementUtil::saveCellArraysBinary(folderPrefix + "asAB" + affix + ".txt", asAB)) return false;
		if (!ElementUtil::saveCellArraysBinary(folderPrefix + "asAC" + affix + ".txt", asAC)) return false;
		if (!ElementUtil::saveCellArraysBinary(folderPrefix + "atAB" + affix + ".txt", atAB)) return false;
		if (!ElementUtil::saveCellArraysBinary(folderPrefix + "atAC" + affix + ".txt", atAC)) return false;
		if (!ElementUtil::saveCellArrayBinary(folderPrefix + "usAB" + affix + ".txt", usAB)) return false;
		if (!ElementUtil::saveCellArrayBinary(folderPrefix + "usAC" + affix + ".txt", usAC)) return false;
		if (!ElementUtil::saveCellArrayBinary(folderPrefix + "utAB" + affix + ".txt", utAB)) return false;
		if (!ElementUtil::saveCellArrayBinary(folderPrefix + "utAC" + affix + ".txt", utAC)) return false;
	}

	return true;
}

void DemoIO::error(string s) {
	cout << "Error: " << s << endl;
	system("pause");
}