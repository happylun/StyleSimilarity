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

#include <Eigen/Eigen>

#include "IO/BaseIO.h"

using namespace std;

namespace StyleSimilarity {

	class DemoIO : public BaseIO {

	public:

		static bool process();

	private:

		static bool runSinglePipeline();
		static bool runPairwisePipeline();
		static bool runTripletPipeline();

		static bool runSingleModelPreprocessing(string meshName);
		static bool runSingleModelSegmentation(string meshName);
		static bool runSingleModelFeature(string meshName);
		static bool runSingleModelSaliency(int shapeID, string meshName);

		static bool runPairedModelVoting(string sourceMeshName, string targetMeshName);
		static bool runPairedModelDistance(string sourceMeshName, string targetMeshName);

		static bool runTripletDistance(int tripletID, string meshA, string meshB, string meshC);

		static void error(string s);
	};

}