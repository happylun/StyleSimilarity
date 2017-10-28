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

#include "ClusterKMeans.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace StyleSimilarity;

bool ClusterKMeans::cluster(int numClusters, Eigen::MatrixXd &inPoints, Eigen::VectorXi &outIndices) {

	int numPoints = (int)inPoints.rows();
	int numDimensions = (int)inPoints.cols();

	if (numClusters > numPoints) {
		cout << "Error: incorrect number of clusters" << endl;
		return false;
	}

	// choose initial means
	Eigen::MatrixXd means(numClusters, numDimensions);
	/*
	if (true) { // random points
		vector<int> indices(numPoints);
		for (int j = 0; j < numPoints; j++) indices[j] = j;
		random_shuffle(indices.begin(), indices.end());
#pragma omp parallel for
		for (int j = 0; j < numClusters; j++) {
			means.row(j) = inPoints.row(indices[j]);
		}
	}
	*/
	if (true) { // furthest point heuristic
		default_random_engine rng;
		uniform_int_distribution<int> uni(0, numPoints - 1);
		int index = uni(rng);
		means.row(0) = inPoints.row(index); // randomly choose first point
		Eigen::VectorXd currentDist = (inPoints.rowwise() - means.row(0)).rowwise().squaredNorm();
		for (int k = 1; k < numClusters; k++) {
			currentDist.maxCoeff(&index);
			means.row(k) = inPoints.row(index);
			currentDist = (inPoints.rowwise() - means.row(k)).rowwise().squaredNorm().cwiseMin(currentDist);
		}
	}

	Eigen::VectorXi clusterIndices(numPoints);
	clusterIndices.setZero();

	// main loop
	cout << "Iterating K-means" << endl;
	int iterCount = 0;
	while (true) {

		// assign points to means
		Eigen::VectorXi newIndices(numPoints);
#pragma omp parallel for
		for (int pointID = 0; pointID < numPoints; pointID++) {
			Eigen::VectorXd dists = (means.rowwise() - inPoints.row(pointID)).rowwise().norm();
			int index;
			dists.minCoeff(&index, (int*)0);
			newIndices[pointID] = index;
		}
		if (newIndices == clusterIndices) break; // converged
		clusterIndices.swap(newIndices);

		// compute new means
#pragma omp parallel for
		for (int clusterID = 0; clusterID < numClusters; clusterID++) {
			Eigen::VectorXd newMean(numDimensions);
			newMean.setZero();
			int pointCount = 0;
			for (int pointID = 0; pointID < numPoints; pointID++) {
				if (clusterIndices[pointID] != clusterID) continue;
				newMean += inPoints.row(pointID);
				pointCount++;
			}
			if (pointCount) newMean.array() *= 1.0 / pointCount;
			means.row(clusterID) = newMean;
		}
		iterCount++;
	}
	cout << "Converged at " << iterCount << " iterations" << endl;

	//cout << means << endl;

	outIndices.swap(clusterIndices);

	return true;
}

bool ClusterKMeans::test() {

	cout << "K-means test" << endl;

	cout << "Generating data..." << endl;

	default_random_engine rng;
	normal_distribution<double> p1X(1.0, 0.5);
	normal_distribution<double> p1Y(1.0, 0.5);
	normal_distribution<double> p2X(-1.0, 0.3);
	normal_distribution<double> p2Y(3.0, 0.3);
	normal_distribution<double> p3X(0.0, 1.0);
	normal_distribution<double> p3Y(-2.0, 1.0);
	normal_distribution<double> p4X(3.0, 0.8);
	normal_distribution<double> p4Y(3.0, 0.8);

	Eigen::MatrixXd data(40000, 2);
	for (int i = 0; i<10000; i++) {
		data(i * 4, 0) = p1X(rng);
		data(i * 4, 1) = p1Y(rng);
		data(i * 4 + 1, 0) = p2X(rng);
		data(i * 4 + 1, 1) = p2Y(rng);
		data(i * 4 + 2, 0) = p3X(rng);
		data(i * 4 + 2, 1) = p3Y(rng);
		data(i * 4 + 3, 0) = p4X(rng);
		data(i * 4 + 3, 1) = p4Y(rng);
	}

	cout << "Clustering..." << endl;

	Eigen::VectorXi indices;
	if (!cluster(4, data, indices)) return false;

	return true;
}