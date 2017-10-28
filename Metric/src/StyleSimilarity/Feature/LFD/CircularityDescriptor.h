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

namespace LFD {
	class CircularityDescriptor {

	private:

		CircularityDescriptor() {}
		~CircularityDescriptor() {}

	public:

		static bool calculate(
			vector<unsigned char> &inImage,
			double &outDescriptor);

	private:

		static double calculateArea(vector<unsigned char> &inImage);
		static double calculatePerimeter(vector<unsigned char> &inImage);
	};
}