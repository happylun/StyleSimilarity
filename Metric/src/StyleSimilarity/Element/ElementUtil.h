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

#include "Eigen/Eigen"

#include "Data/StyleSimilarityTypes.h"

using namespace std;

namespace StyleSimilarity {

	class ElementUtil {

	private:

		// make it non-instantiable
		ElementUtil() {}
		~ElementUtil() {}

	public:

		static bool convertAffineToTRS(Eigen::Affine3d &inAffine, vector<double> &outTRS);
		//static bool convertTRSToAffine(vector<double> &inTRS, Eigen::Affine3d &outAffine);

		static bool saveVectorASCII(string fileName, Eigen::VectorXd &inVector);
		static bool loadVectorASCII(string fileName, Eigen::VectorXd &outVector);

		static bool saveRowVectorASCII(string fileName, Eigen::RowVectorXd &inVector);
		static bool loadRowVectorASCII(string fileName, Eigen::RowVectorXd &outVector);

		static bool saveMatrixASCII(string fileName, Eigen::MatrixXd &inMatrix);
		static bool loadMatrixASCII(string fileName, Eigen::MatrixXd &outMatrix);

		static bool saveVectorBinary(string fileName, Eigen::VectorXd &inVector);
		static bool loadVectorBinary(string fileName, Eigen::VectorXd &outVector);

		static bool saveRowVectorBinary(string fileName, Eigen::RowVectorXd &inVector);
		static bool loadRowVectorBinary(string fileName, Eigen::RowVectorXd &outVector);

		static bool saveMatrixBinary(string fileName, Eigen::MatrixXd &inMatrix);
		static bool loadMatrixBinary(string fileName, Eigen::MatrixXd &outMatrix);

		static bool saveCellArrayBinary(string fileName, vector<int> &inArray);
		static bool loadCellArrayBinary(string fileName, vector<int> &outArray);

		static bool saveCellArraysBinary(string fileName, vector<vector<int>> &inArrays);
		static bool loadCellArraysBinary(string fileName, vector<vector<int>> &outArrays);
	};
}