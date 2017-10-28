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

using namespace std;

class ARPACKHelper {

private:

	ARPACKHelper() {}
	~ARPACKHelper() {}

public:

	struct TTriplet {
		int row, col;
		double value;
		TTriplet() {}
		TTriplet(int r, int c, double v) : row(r), col(c), value(v) {}
	};

	static bool compute(
		int inNumDim, int inNumEV,
		vector<TTriplet> &inMatrix,
		vector<double> &outValues,
		vector<vector<double>> &outVectors,
		double eps = 0.0);

private:

	static bool product(int n, vector<TTriplet> &mat, double *in, double*out);
};