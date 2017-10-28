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

#include "FeatureAsset.h"

#include <iostream>
#include <fstream>

using namespace StyleSimilarity;

FeatureAsset::FeatureAsset() {
}

FeatureAsset::~FeatureAsset() {
}

bool FeatureAsset::loadCurvature(string fileName) {
	return FeatureCurvature::loadFeature(fileName, mCurvature);
}
bool FeatureAsset::loadSDF(string fileName) {
	return loadFeature(fileName, mSDF);
}
bool FeatureAsset::loadGeodesic(string fileName) {
	return loadFeature(fileName, mGeodesic);
}
bool FeatureAsset::loadAO(string fileName) {
	return loadFeature(fileName, mAO);
}
bool FeatureAsset::loadSD(string fileName) {
	return loadFeature(fileName, mSD);
}
bool FeatureAsset::loadLFD(string fileName) {
	return loadFeature(fileName, mLFD);
}
bool FeatureAsset::loadCurve(string fileNamePrefix) {
	for (int curveID = 0; curveID < CurveRidgeValley::CURVE_TYPES; curveID++) {
		auto &curve = mCurve[curveID];
		string fileName = fileNamePrefix + "-" + to_string(curveID) + ".ply";
		if (!PlyLoader::loadPoint(fileName, &curve.positions, &curve.normals)) return false;
		curve.amount = (int)curve.positions.size();
		if (!SampleUtil::buildKdTree(curve.positions, mCurveTree[curveID], mCurveTreeData[curveID])) return false;
	}
	return true;
}
bool FeatureAsset::loadTalFPFH(string fileName) {
	return FeatureTalSaliency::loadFeature(fileName, mTalFPFH);
}
bool FeatureAsset::loadTalSI(string fileName) {
	return FeatureTalSaliency::loadFeature(fileName, mTalSI);
}
bool FeatureAsset::loadTalSC(string fileName) {
	return FeatureTalSaliency::loadFeature(fileName, mTalSC);
}
bool FeatureAsset::loadSaliency(string fileName) {
	return ElementUtil::loadMatrixBinary(fileName, mSaliency);
}

bool FeatureAsset::visualizeCurvature(string fileName, TSampleSet &samples) {
	FeatureCurvature feature(&samples, &mCurvature);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeSDF(string fileName, TSampleSet &samples) {
	FeatureSDF feature(&samples, 0, &mSDF);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeGeodesic(string fileName, TSampleSet &samples) {
	FeatureGeodesic feature(&samples, &mGeodesic);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeAO(string fileName, TSampleSet &samples) {
	FeatureAmbientOcclusion feature(&samples, 0, &mAO);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeSD(string fileName) {
	FeatureShapeDistributions feature(0, &mSD);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeTalFPFH(string fileName, TSampleSet &samples) {
	FeatureTalSaliency feature(&samples, &mTalFPFH);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeTalSI(string fileName, TSampleSet &samples) {
	FeatureTalSaliency feature(&samples, &mTalSI);
	return feature.visualize(fileName);
}
bool FeatureAsset::visualizeTalSC(string fileName, TSampleSet &samples) {
	FeatureTalSaliency feature(&samples, &mTalSC);
	return feature.visualize(fileName);
}

bool FeatureAsset::saveFeature(string fileName, vector<double> &feature) {

	ofstream outFile(fileName, ios::binary);
	if (!outFile.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}

	for (auto value : feature) {
		outFile.write((const char*)(&value), sizeof(value));
	}
	outFile.close();

	return true;
}


bool FeatureAsset::loadFeature(string fileName, vector<double> &feature) {

	ifstream inFile(fileName, ios::binary | ios::ate);
	if (!inFile.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	std::streampos fileSize = inFile.tellg();
	int n = (int)(fileSize / sizeof(feature[0]));
	feature.resize(n);

	inFile.seekg(0, ios::beg);
	for (int d = 0; d < n; d++) {
		inFile.read((char*)(&feature[d]), sizeof(feature[0]));
	}
	inFile.close();

	return true;
}
