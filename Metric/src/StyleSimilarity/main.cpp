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

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#include "Eigen/Core"

#include "Data/StyleSimilarityConfig.h"

#include "IO/Run/DemoIO.h"

#include "Utility/Timer.h"

using namespace std;
using namespace StyleSimilarity;

int main(int argc, char** argv) {

	Timer::tic();

	// init
	Eigen::initParallel();
	srand((unsigned int)time(0));
	
	// load all configs
	for(int cfgID=1; cfgID<argc; cfgID++) {
		if (!StyleSimilarityConfig::loadConfig(argv[cfgID])) return -1;
	}

#ifdef _OPENMP
	// limit number of threads
	if (StyleSimilarityConfig::mPipeline_MaximumThreads) {
		omp_set_num_threads(StyleSimilarityConfig::mPipeline_MaximumThreads);
	} else {
		StyleSimilarityConfig::mPipeline_MaximumThreads = omp_get_max_threads();
	}
#endif

	// run pipeline
	if (StyleSimilarityConfig::mPipeline_RunDemo && !DemoIO::process()) return -1;

	cout << "Total time: " << Timer::toString(Timer::toc()) << endl;
	system("pause");

	return 0;
}
