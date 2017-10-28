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

#include <vector>
#include <string>

#include <Eigen/Eigen>

#include "Library/SidKDTreeHelper.h"

#include "Feature/FeatureCurvature.h"
#include "Feature/FeatureSDF.h"
#include "Feature/FeatureLFD.h"
#include "Feature/FeatureGeodesic.h"
#include "Feature/FeatureAmbientOcclusion.h"
#include "Feature/FeatureShapeDistributions.h"
#include "Feature/FeatureFPFH.h"
#include "Feature/FeatureSpinImages.h"
#include "Feature/FeatureShapeContexts.h"
#include "Feature/FeatureTalSaliency.h"
#include "Curve/CurveRidgeValley.h"

#include "Feature/FeatureUtil.h"
#include "Sample/SampleUtil.h"
#include "Element/ElementUtil.h"

#include "Utility/PlyLoader.h"

using namespace std;

namespace StyleSimilarity {

	class FeatureAsset {

		friend class ElementMetric;
		friend class ElementVisualization;

	public:

		FeatureAsset();
		~FeatureAsset();

		static bool saveFeature(string fileName, vector<double> &feature);
		static bool loadFeature(string fileName, vector<double> &feature);

	public:

		bool loadCurvature(string fileName);
		bool loadSDF(string fileName);
		bool loadGeodesic(string fileName);
		bool loadAO(string fileName);
		bool loadSD(string fileName);
		bool loadLFD(string fileName);
		bool loadCurve(string fileNamePrefix);
		bool loadTalFPFH(string fileName);
		bool loadTalSI(string fileName);
		bool loadTalSC(string fileName);
		bool loadSaliency(string fileName);

		bool visualizeCurvature(string fileName, TSampleSet &samples);
		bool visualizeSDF(string fileName, TSampleSet &samples);
		bool visualizeGeodesic(string fileName, TSampleSet &samples);
		bool visualizeAO(string fileName, TSampleSet &samples);
		bool visualizeSD(string fileName);
		bool visualizeTalFPFH(string fileName, TSampleSet &samples);
		bool visualizeTalSI(string fileName, TSampleSet &samples);
		bool visualizeTalSC(string fileName, TSampleSet &samples);

	protected:

		vector<FeatureCurvature::TCurvature> mCurvature;
		vector<double> mSDF;
		vector<double> mGeodesic;
		vector<double> mAO;
		vector<double> mSD;
		vector<double> mLFD;
		TPointSet mCurve[CurveRidgeValley::CURVE_TYPES];
		SKDTree mCurveTree[CurveRidgeValley::CURVE_TYPES];
		SKDTreeData mCurveTreeData[CurveRidgeValley::CURVE_TYPES];
		vector<vector<double>> mTalFPFH; // # metrics : # points
		vector<vector<double>> mTalSI; // # metrics : # points
		vector<vector<double>> mTalSC; // # metrics : # points

	public:

		Eigen::MatrixXd mSaliency; // # points X dimS
	};
}