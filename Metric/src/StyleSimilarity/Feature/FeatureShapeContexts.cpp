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

#include "FeatureShapeContexts.h"

#include <fstream>
#include <iostream>

#include "Sample/SampleUtil.h"

#include "Data/StyleSimilarityConfig.h"

#include "Feature/FeatureUtil.h"

#include "Utility/PlyExporter.h"

using namespace StyleSimilarity;

const int NUM_RADIUS_BINS = 5;
const int NUM_ANGLE_BINS = 4;
const int NUM_BINS = NUM_RADIUS_BINS * NUM_ANGLE_BINS;

//#define VISUALIZE_BINS 12345

bool FeatureShapeContexts::calculate(
	TSampleSet &samples,
	TSampleSet &points,
	vector<vector<double>> &features,
	double radius)
{
	static vector<double> radiusBins; // make it static so that it can be re-used in later calls
	if (radiusBins.empty()) {
		const double p = 1.0; // UNDONE: param radius bin size power factor
		radiusBins.resize(NUM_RADIUS_BINS);
		double s = (double)(NUM_RADIUS_BINS + 1);
		for (int r = 0; r < NUM_RADIUS_BINS; r++) {
			radiusBins[r] = pow(log(s / (s - r - 1)) / log(s), p);
		}
	}

	vec3 bbMin, bbMax;
	if (!SampleUtil::computeAABB(samples, bbMin, bbMax)) return false;
	float shapeScale = (bbMax - bbMin).length();

	vector<vec3i> binColors(NUM_BINS);
	for (int binID = 0; binID < NUM_BINS; binID++) {
		binColors[binID] = vec3i(cml::random_integer(0, 255), cml::random_integer(0, 255), cml::random_integer(0, 255));
	}
	
	features.resize(samples.amount);
#ifndef VISUALIZE_BINS
#pragma omp parallel for
#endif
	for (int sampleID = 0; sampleID < samples.amount; sampleID++) {
		vec3 sampleP = samples.positions[sampleID];
		vec3 sampleN = samples.normals[sampleID];

#ifdef VISUALIZE_BINS
		vector<vec3i> scColors(points.amount);
		//int scPointID = VISUALIZE_BINS;
		int scPointID = -1;
		if (scPointID < 0 && sampleP[1] > 0.49f && fabs(sampleP[0] - 0.05f) < 0.01f && fabs(sampleP[2] - 0.05f) < 0.01f) {
			scPointID = sampleID;
		}
#endif

		vector<double> hist(NUM_BINS, 0);
		for (int pointID = 0; pointID < points.amount; pointID++) {
			vec3 pointP = points.positions[pointID];
			double radius = (pointP - sampleP).length() / shapeScale; // [0, 1]
			double angle = cml::unsigned_angle(vec3d(pointP-sampleP), vec3d(sampleN)); // [0, pi]
			int binR = 0;
			for (; binR < NUM_RADIUS_BINS; binR++) if (radius <= radiusBins[binR]) break;
			int binA = (int)(angle / cml::constantsd::pi() * NUM_ANGLE_BINS);
			binR = cml::clamp(binR, 0, NUM_RADIUS_BINS - 1);
			binA = cml::clamp(binA, 0, NUM_ANGLE_BINS - 1);
			int binPos = binA * NUM_RADIUS_BINS + binR;
			double weight = 1 / (radius + 0.1);
			hist[binPos] += weight;

#ifdef VISUALIZE_BINS
			if (sampleID == scPointID) {
				scColors[pointID] = binColors[binPos];
			}
#endif
		}

		auto &feature = features[sampleID];
		feature.resize(NUM_BINS);
		double sumHist = 0;
		for (int binID = 0; binID < NUM_BINS; binID++) sumHist += hist[binID];
		for (int binID = 0; binID < NUM_BINS; binID++) {
			feature[binID] = sumHist ? hist[binID] / sumHist : 0;
		}

#ifdef VISUALIZE_BINS
		if (sampleID == scPointID) {

			vector<vec3> lines;
			for (int j = 0; j < 100; j++) {
				lines.push_back(sampleP + sampleN*j*samples.radius*0.1f);
			}

			PlyExporter pe;
			pe.addPoint(&points.positions, &points.normals, &scColors);
			pe.addPoint(&lines, 0, vec3(0.0f,0.0f,0.0f), vec3i(255,0,0));
			pe.output("Style/2.feature/saliency/SC.ply");
			system("pause");
		}
#endif
	}

	return true;
}
