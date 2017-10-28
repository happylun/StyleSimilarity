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

#include "StyleSimilarityConfig.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

using namespace StyleSimilarity;

// default configuration parameters

bool   StyleSimilarityConfig::mPipeline_DebugPipeline     = false;
bool   StyleSimilarityConfig::mPipeline_DebugSynthesis    = false;
bool   StyleSimilarityConfig::mPipeline_DebugSaliency     = false;
bool   StyleSimilarityConfig::mPipeline_DebugCurve        = false;
bool   StyleSimilarityConfig::mPipeline_DebugClustering   = false;
bool   StyleSimilarityConfig::mPipeline_DebugFeature      = false;
bool   StyleSimilarityConfig::mPipeline_DebugSegmentation = false;
bool   StyleSimilarityConfig::mPipeline_DebugMatching     = false;
bool   StyleSimilarityConfig::mPipeline_DebugAnything     = false;
bool   StyleSimilarityConfig::mPipeline_TestFeature       = false;
bool   StyleSimilarityConfig::mPipeline_TestSymmetry      = false;
bool   StyleSimilarityConfig::mPipeline_RunDemo           = false;
bool   StyleSimilarityConfig::mPipeline_RunDistance       = false;
bool   StyleSimilarityConfig::mPipeline_RunCoSeg          = false;
int    StyleSimilarityConfig::mPipeline_MaximumThreads    = 0;
int    StyleSimilarityConfig::mPipeline_Stage             = 0;

string StyleSimilarityConfig::mData_DataSetRootFolder = "";
string StyleSimilarityConfig::mData_CustomString1     = "";
string StyleSimilarityConfig::mData_CustomString2     = "";
string StyleSimilarityConfig::mData_CustomString3     = "";
double StyleSimilarityConfig::mData_CustomNumber1     = 0;
double StyleSimilarityConfig::mData_CustomNumber2     = 0;
double StyleSimilarityConfig::mData_CustomNumber3     = 0;

int    StyleSimilarityConfig::mSegment_SpectralClusters             = 200;
int    StyleSimilarityConfig::mSegment_NParVisibilitySampleNumber   = 10;
TDList StyleSimilarityConfig::mSegment_NParVisibilityThresholdList  = TDList();
double StyleSimilarityConfig::mSegment_NParCoplanarAngularThreshold = 0.01;
double StyleSimilarityConfig::mSegment_NParPruningAreaRatio         = 0.01;
bool   StyleSimilarityConfig::mSegment_NParSDFMerging               = true;
bool   StyleSimilarityConfig::mSegment_NParOutputLastResult         = false;

int    StyleSimilarityConfig::mSample_WholeMeshSampleNumber   = 20000;
double StyleSimilarityConfig::mSample_MinimumSampleRate       = 0.8;
int    StyleSimilarityConfig::mSample_MaximumFailedCount      = 10000;
int    StyleSimilarityConfig::mSample_MaximumCheckedFaceCount = 100000;
bool   StyleSimilarityConfig::mSample_VisibilityChecking      = true;
bool   StyleSimilarityConfig::mSample_AddVirtualGround        = false;

double StyleSimilarityConfig::mCurvature_PatchGeodesicRadius    = 5.0;
double StyleSimilarityConfig::mCurvature_NeighborGaussianRadius = 2.0;
double StyleSimilarityConfig::mCurvature_MaxMagnitudeRadius     = 0.05;

double StyleSimilarityConfig::mCurve_RVStrengthThreshold        = 0.0;
double StyleSimilarityConfig::mCurve_RVLengthThreshold          = 0.05;
double StyleSimilarityConfig::mCurve_RVPointSamplingRadius      = 0.001;

double StyleSimilarityConfig::mMatch_RejectDistanceThreshold      = 5.0;

double StyleSimilarityConfig::mCluster_BandwidthCumulatedWeight         = 0.01;
int    StyleSimilarityConfig::mCluster_MeanShiftMaxIterations           = 10000;
double StyleSimilarityConfig::mCluster_MeanShiftConvergenceThreshold    = 0.0001;
double StyleSimilarityConfig::mCluster_ModeMergingThreshold             = 1.0;

TDList StyleSimilarityConfig::mOptimization_DistanceSigmaList            = TDList();
double StyleSimilarityConfig::mOptimization_DistanceSigmaPercentile      = 0.5;
double StyleSimilarityConfig::mOptimization_MegaSigmaFactor              = 1.0;
double StyleSimilarityConfig::mOptimization_UnmatchedUnaryFactor         = 5.0;
double StyleSimilarityConfig::mOptimization_MatchedPatchCoverage         = 0.5;
bool   StyleSimilarityConfig::mOptimization_FirstIteration               = true;
int    StyleSimilarityConfig::mOptimization_ExpansionIterationNumber     = 10000;

bool StyleSimilarityConfig::loadConfig(string fileName) {

	ifstream cfgFile(fileName);
	if(!cfgFile.is_open()) {
		cout << "Error: cannot load config file " << fileName << endl;
		return false;
	}

	string category;
	while(!cfgFile.eof()) {

		string line;
		getline(cfgFile, line);
		line = trim(line);

		if(line.length() == 0) continue;
		if(line.substr(0,2) == "//") continue;

		if(line.front() == '[' && line.back() == ']') {
			category = trim(line.substr(1, line.length()-2));
			continue;
		}
		
		int pos = (int)line.find_first_of('=');
		string key = trim(line.substr(0, pos));
		string value = trim(line.substr(pos+1));

		if(category == "PIPELINE") {

			if (key == "Debug Pipeline") {
				mPipeline_DebugPipeline = parseBool(value);
			} else if (key == "Debug Synthesis") {
				mPipeline_DebugSynthesis = parseBool(value);
			} else if (key == "Debug Saliency") {
				mPipeline_DebugSaliency = parseBool(value);
			} else if (key == "Debug Curve") {
				mPipeline_DebugCurve = parseBool(value);
			} else if (key == "Debug Clustering") {
				mPipeline_DebugClustering = parseBool(value);
			} else if (key == "Debug Feature") {
				mPipeline_DebugFeature = parseBool(value);
			} else if( key == "Debug Segmentation" ) {
				mPipeline_DebugSegmentation = parseBool(value);
			} else if( key == "Debug Matching" ) {
				mPipeline_DebugMatching = parseBool(value);
			} else if (key == "Debug Anything") {
				mPipeline_DebugAnything = parseBool(value);
			} else if (key == "Test Feature") {
				mPipeline_TestFeature = parseBool(value);
			} else if (key == "Test Symmetry") {
				mPipeline_TestSymmetry= parseBool(value);
			} else if (key == "Run Demo") {
				mPipeline_RunDemo = parseBool(value);
			} else if (key == "Run Distance") {
				mPipeline_RunDistance = parseBool(value);
			} else if (key == "Run Co-Seg") {
				mPipeline_RunCoSeg = parseBool(value);
			} else if (key == "Maximum Threads") {
				mPipeline_MaximumThreads = parseInt(value);
			} else if (key == "Stage") {
				mPipeline_Stage = parseInt(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if (category == "DATA") {

			if (key == "Data Set Root Folder") {
				mData_DataSetRootFolder = value;
			} else if (key == "Custom String 1") {
				mData_CustomString1 = value;
			} else if (key == "Custom String 2") {
				mData_CustomString2 = value;
			} else if (key == "Custom String 3") {
				mData_CustomString3 = value;
			} else if (key == "Custom Number 1") {
				mData_CustomNumber1 = parseDouble(value);
			} else if (key == "Custom Number 2") {
				mData_CustomNumber2 = parseDouble(value);
			} else if (key == "Custom Number 3") {
				mData_CustomNumber3 = parseDouble(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if(category == "SEGMENT") {

			if (key == "Spectral Clusters") {
				mSegment_SpectralClusters = parseInt(value);
			} else if (key == "NPar Visibility Sample Number") {
				mSegment_NParVisibilitySampleNumber = parseInt(value);
			} else if ( key == "NPar Visibility Threshold List" ) {
				mSegment_NParVisibilityThresholdList = parseDoubleList(value);
			} else if ( key == "NPar Coplanar Angular Threshold" ) {
				mSegment_NParCoplanarAngularThreshold = parseDouble(value);
			} else if ( key == "NPar Pruning Area Ratio" ) {
				mSegment_NParPruningAreaRatio = parseDouble(value);
			} else if (key == "NPar SDF Merging") {
				mSegment_NParSDFMerging = parseBool(value);
			} else if (key == "NPar Output Last Result") {
				mSegment_NParOutputLastResult = parseBool(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if(category == "SAMPLE") {

			if( key == "Whole Mesh Sample Number" ) {
				mSample_WholeMeshSampleNumber = parseInt(value);			
			} else if( key == "Minimum Success Rate" ) {
				mSample_MinimumSampleRate = parseDouble(value);
			} else if( key == "Maximum Failed Count" ) {
				mSample_MaximumFailedCount = parseInt(value);
			} else if (key == "Maximum Checked Face Count") {
				mSample_MaximumCheckedFaceCount = parseInt(value);
			} else if (key == "Visibility Checking") {
				mSample_VisibilityChecking = parseBool(value);
			} else if (key == "Add Virtual Ground") {
				mSample_AddVirtualGround = parseBool(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if (category == "CURVATURE") {

			if (key == "Patch Geodesic Radius") {
				mCurvature_PatchGeodesicRadius = parseDouble(value);
			} else if (key == "Neighbor Gaussian Radius") {
				mCurvature_NeighborGaussianRadius = parseDouble(value);
			} else if (key == "Max Magnitude Radius") {
				mCurvature_MaxMagnitudeRadius = parseDouble(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if (category == "CURVE") {

			if (key == "RV Strength Threshold") {
				mCurve_RVStrengthThreshold = parseDouble(value);
			} else if (key == "RV Length Threshold") {
				mCurve_RVLengthThreshold = parseDouble(value);
			} else if (key == "RV Point Sampling Radius") {
				mCurve_RVPointSamplingRadius = parseDouble(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if(category == "MATCH") {

			if( key == "Reject Distance Threshold" ) {
				mMatch_RejectDistanceThreshold = parseDouble(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if(category == "CLUSTER") {

			if (key == "Bandwidth Cumulated Weight") {
				mCluster_BandwidthCumulatedWeight = parseDouble(value);
			} else if( key == "Mean Shift Max Iterations" ) {
				mCluster_MeanShiftMaxIterations = parseInt(value);
			} else if( key == "Mean Shift Convergence Threshold" ) {
				mCluster_MeanShiftConvergenceThreshold = parseDouble(value);
			} else if( key == "Mode Merging Threshold" ) {
				mCluster_ModeMergingThreshold = parseDouble(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else if(category == "OPTIMIZATION") {

			if( key == "Distance Sigma List" ) {
				mOptimization_DistanceSigmaList = parseDoubleList(value);
			} else if (key == "Distance Sigma Percentile") {
				mOptimization_DistanceSigmaPercentile = parseDouble(value);
			} else if (key == "Mega Sigma Factor") {
				mOptimization_MegaSigmaFactor = parseDouble(value);
			} else if (key == "Unmatched Unary Factor") {
				mOptimization_UnmatchedUnaryFactor = parseDouble(value);
			} else if (key == "Matched Patch Coverage") {
				mOptimization_MatchedPatchCoverage = parseDouble(value);
			} else if (key == "First Iteration") {
				mOptimization_FirstIteration = parseBool(value);
			} else if( key == "Expansion Iteration Number" ) {
				mOptimization_ExpansionIterationNumber = parseInt(value);
			} else {
				cout << "Error: unrecognized config => " << line << endl;
			}

		} else {
			cout << "Error: unrecognized config => " << line << endl;
		}

	}

	cfgFile.close();

	return true;
}

bool StyleSimilarityConfig::saveConfig(string fileName) {

	ofstream cfgFile(fileName);

	cfgFile << endl << "[PIPELINE]" << endl << endl;

	cfgFile << "Debug Pipeline"        << " = " << (mPipeline_DebugPipeline     ? "true" : "false") << endl;
	cfgFile << "Debug Synthesis"       << " = " << (mPipeline_DebugSynthesis    ? "true" : "false") << endl;
	cfgFile << "Debug Saliency"        << " = " << (mPipeline_DebugSaliency     ? "true" : "false") << endl;
	cfgFile << "Debug Curve"           << " = " << (mPipeline_DebugCurve        ? "true" : "false") << endl;
	cfgFile << "Debug Clustering"      << " = " << (mPipeline_DebugClustering   ? "true" : "false") << endl;
	cfgFile << "Debug Feature"         << " = " << (mPipeline_DebugFeature      ? "true" : "false") << endl;
	cfgFile << "Debug Segmentation"    << " = " << (mPipeline_DebugSegmentation ? "true" : "false") << endl;
	cfgFile << "Debug Matching"        << " = " << (mPipeline_DebugMatching     ? "true" : "false") << endl;
	cfgFile << "Debug Anything"        << " = " << (mPipeline_DebugAnything     ? "true" : "false") << endl;
	cfgFile << "Test Feature"          << " = " << (mPipeline_TestFeature       ? "true" : "false") << endl;
	cfgFile << "Test Symmetry"         << " = " << (mPipeline_TestSymmetry      ? "true" : "false") << endl;
	cfgFile << "Run Demo"              << " = " << (mPipeline_RunDemo           ? "true" : "false") << endl;
	cfgFile << "Run Distance"          << " = " << (mPipeline_RunDistance       ? "true" : "false") << endl;
	cfgFile << "Run Co-Seg"            << " = " << (mPipeline_RunCoSeg          ? "true" : "false") << endl;
	cfgFile << "Maximum Threads"       << " = " << mPipeline_MaximumThreads << endl;
	cfgFile << "Stage"                 << " = " << mPipeline_Stage << endl;

	cfgFile << endl << "[DATA]" << endl << endl;

	cfgFile << "Data Set Root Folder" << " = " << mData_DataSetRootFolder << endl;
	cfgFile << "Custom String 1"      << " = " << mData_CustomString1 << endl;
	cfgFile << "Custom String 2"      << " = " << mData_CustomString2 << endl;
	cfgFile << "Custom String 3"      << " = " << mData_CustomString3 << endl;
	cfgFile << "Custom Number 1"      << " = " << mData_CustomNumber1 << endl;
	cfgFile << "Custom Number 2"      << " = " << mData_CustomNumber2 << endl;
	cfgFile << "Custom Number 3"      << " = " << mData_CustomNumber3 << endl;

	cfgFile << endl << "[SEGMENT]" << endl << endl;

	cfgFile << "Spectral Clusters"               << " = " << mSegment_SpectralClusters << endl;
	cfgFile << "NPar Visibility Sample Number"   << " = " << mSegment_NParVisibilitySampleNumber << endl;
	cfgFile << "NPar Visibility Threshold List"  << " = " << mSegment_NParVisibilityThresholdList << endl;
	cfgFile << "NPar Coplanar Angular Threshold" << " = " << mSegment_NParCoplanarAngularThreshold << endl;
	cfgFile << "NPar Pruning Area Ratio"         << " = " << mSegment_NParPruningAreaRatio << endl;
	cfgFile << "NPar SDF Merging"                << " = " << (mSegment_NParSDFMerging ? "true" : "false") << endl;
	cfgFile << "NPar Output Last Result"         << " = " << (mSegment_NParOutputLastResult ? "true" : "false") << endl;

	cfgFile << endl << "[SAMPLE]" << endl << endl;

	cfgFile << "Whole Mesh Sample Number"   << " = " << mSample_WholeMeshSampleNumber << endl;
	cfgFile << "Minimum Success Rate"       << " = " << mSample_MinimumSampleRate << endl;
	cfgFile << "Maximum Failed Count"       << " = " << mSample_MaximumFailedCount << endl;
	cfgFile << "Maximum Checked Face Count" << " = " << mSample_MaximumCheckedFaceCount << endl;
	cfgFile << "Visibility Checking"        << " = " << (mSample_VisibilityChecking ? "true" : "false") << endl;
	cfgFile << "Add Virtual Ground"         << " = " << (mSample_AddVirtualGround ? "true" : "false") << endl;

	cfgFile << endl << "[CURVATURE]" << endl << endl;

	cfgFile << "Patch Geodesic Radius"    << " = " << mCurvature_PatchGeodesicRadius << endl;
	cfgFile << "Neighbor Gaussian Radius" << " = " << mCurvature_NeighborGaussianRadius << endl;
	cfgFile << "Max Magnitude Radius"     << " = " << mCurvature_MaxMagnitudeRadius << endl;

	cfgFile << endl << "[CURVE]" << endl << endl;

	cfgFile << "RV Strength Threshold"         << " = " << mCurve_RVStrengthThreshold << endl;
	cfgFile << "RV Length Threshold"           << " = " << mCurve_RVLengthThreshold << endl;
	cfgFile << "RV Point Sampling Radius"      << " = " << mCurve_RVPointSamplingRadius << endl;

	cfgFile << endl << "[MATCH]" << endl << endl;

	cfgFile << "Reject Distance Threshold"      << " = " << mMatch_RejectDistanceThreshold << endl;

	cfgFile << endl << "[CLUSTER]" << endl << endl;

	cfgFile << "Bandwidth Cumulated Weight"        << " = " << mCluster_BandwidthCumulatedWeight << endl;
	cfgFile << "Mean Shift Max Iterations"         << " = " << mCluster_MeanShiftMaxIterations << endl;
	cfgFile << "Mean Shift Convergence Threshold"  << " = " << mCluster_MeanShiftConvergenceThreshold << endl;
	cfgFile << "Mode Merging Threshold"            << " = " << mCluster_ModeMergingThreshold << endl;

	cfgFile << endl << "[OPTIMIZATION]" << endl << endl;

	cfgFile << "Distance Sigma List"         << " = " << mOptimization_DistanceSigmaList << endl;
	cfgFile << "Distance Sigma Percentile"   << " = " << mOptimization_DistanceSigmaPercentile << endl;
	cfgFile << "Mega Sigma Factor"           << " = " << mOptimization_MegaSigmaFactor << endl;
	cfgFile << "Unmatched Unary Factor"      << " = " << mOptimization_UnmatchedUnaryFactor << endl;
	cfgFile << "Matched Patch Coverage"      << " = " << mOptimization_UnmatchedUnaryFactor << endl;
	cfgFile << "First Iteration"             << " = " << (mOptimization_FirstIteration ? "true" : "false") << endl;
	cfgFile << "Expansion Iteration Number"  << " = " << mOptimization_ExpansionIterationNumber << endl;

	cfgFile.close();

	return true;
}

string StyleSimilarityConfig::trim(string s) {

	// ref: http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

int StyleSimilarityConfig::parseInt(string s) {
	stringstream ss(s);
	int v;
	ss >> v;
	return v;
}

double StyleSimilarityConfig::parseDouble(string s) {
	stringstream ss(s);
	double v;
	ss >> v;
	return v;
}

bool StyleSimilarityConfig::parseBool(string s) {
	
	string name = s;
	transform(name.begin(), name.end(), name.begin(), ::tolower);
	if(s == "true") {
		return true;
	} else if(s == "false") {
		return false;
	}
	
	cout << "Error: incorrect boolean value: " << s << endl;
	return false;
}

TIList StyleSimilarityConfig::parseIntList(string s) {

	stringstream ss(s);
	TIList list;
	list.values.clear();
	while(!ss.eof()) {
		int v;
		ss >> v;
		if(!ss.eof() && !ss.good()) {
			cout << "Error: incorrect list \'" << s << "\'" << endl;
		}
		list.values.push_back(v);
	}
	return list;
}

TDList StyleSimilarityConfig::parseDoubleList(string s) {

	stringstream ss(s);
	TDList list;
	list.values.clear();
	while(!ss.eof()) {
		double v;
		ss >> v;
		if(!ss.eof() && !ss.good()) {
			cout << "Error: incorrect list \'" << s << "\'" << endl;
		}
		list.values.push_back(v);
	}
	return list;
}
