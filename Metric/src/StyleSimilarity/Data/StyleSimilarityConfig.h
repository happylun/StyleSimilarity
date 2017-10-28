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

#include <string>
#include <vector>
#include <ostream>

using namespace std;

namespace StyleSimilarity {

	template <class T>
	class TAbstractList {
	public:
		vector<T> values; // variadic template is not yet supported... no elegant initialization...
		friend class StyleSimilarityConfig;
		friend ostream& operator<< (ostream& stream, const TAbstractList<T>& list) {
			for(auto &value : list.values) stream << value << " ";
			return stream;
		}
	};

	typedef TAbstractList<int>    TIList;
	typedef TAbstractList<double> TDList;

	class StyleSimilarityConfig {

	private:

		// make it non-instantiable
		StyleSimilarityConfig() {}
		~StyleSimilarityConfig() {}

	public:

		static bool loadConfig(string fileName);
		static bool saveConfig(string fileName);

	public:

		static bool   mPipeline_DebugPipeline;
		static bool   mPipeline_DebugSynthesis;
		static bool   mPipeline_DebugSaliency;
		static bool   mPipeline_DebugCurve;
		static bool   mPipeline_DebugClustering;
		static bool   mPipeline_DebugFeature;
		static bool   mPipeline_DebugSegmentation;
		static bool   mPipeline_DebugMatching;
		static bool   mPipeline_DebugAnything;
		static bool   mPipeline_TestFeature;
		static bool   mPipeline_TestSymmetry;
		static bool   mPipeline_RunDemo;
		static bool   mPipeline_RunDistance;
		static bool   mPipeline_RunCoSeg;
		static int    mPipeline_MaximumThreads;
		static int    mPipeline_Stage;

		static string mData_DataSetRootFolder;
		static string mData_CustomString1;
		static string mData_CustomString2;
		static string mData_CustomString3;
		static double mData_CustomNumber1;
		static double mData_CustomNumber2;
		static double mData_CustomNumber3;

		static int    mSegment_SpectralClusters;
		static int    mSegment_NParVisibilitySampleNumber;
		static TDList mSegment_NParVisibilityThresholdList;
		static double mSegment_NParCoplanarAngularThreshold;
		static double mSegment_NParPruningAreaRatio;
		static bool   mSegment_NParSDFMerging;
		static bool   mSegment_NParOutputLastResult;

		static int    mSample_WholeMeshSampleNumber;
		static double mSample_MinimumSampleRate;
		static int    mSample_MaximumFailedCount;
		static int    mSample_MaximumCheckedFaceCount;
		static bool   mSample_VisibilityChecking;
		static bool   mSample_AddVirtualGround;

		static double mCurvature_PatchGeodesicRadius;
		static double mCurvature_NeighborGaussianRadius;
		static double mCurvature_MaxMagnitudeRadius;

		static double mCurve_RVStrengthThreshold;
		static double mCurve_RVLengthThreshold;
		static double mCurve_RVPointSamplingRadius;

		static double mMatch_RejectDistanceThreshold;

		static double mCluster_BandwidthCumulatedWeight;
		static int    mCluster_MeanShiftMaxIterations;
		static double mCluster_MeanShiftConvergenceThreshold;
		static double mCluster_ModeMergingThreshold;		

		static TDList mOptimization_DistanceSigmaList;
		static double mOptimization_DistanceSigmaPercentile;
		static double mOptimization_MegaSigmaFactor;
		static double mOptimization_UnmatchedUnaryFactor;
		static double mOptimization_MatchedPatchCoverage;
		static bool   mOptimization_FirstIteration;
		static int    mOptimization_ExpansionIterationNumber;

	private:

		static string trim(string s);

		static int parseInt(string s);
		static double parseDouble(string s);
		static bool parseBool(string s);
		static TIList parseIntList(string s);
		static TDList parseDoubleList(string s);

	};

}