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

#include "ClusterSparseSpectral.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include "Library/ARPACKHelper.h"

#include "Cluster/ClusterKMeans.h"

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

bool ClusterSparseSpectral::cluster(Eigen::SparseMatrix<double> &inGraph, Eigen::VectorXi &outCluster, double eps) {

	if (inGraph.rows() != inGraph.cols()) {
		cout << "Error: not a square matrix" << endl;
		return false;
	}

	int numClusters = StyleSimilarityConfig::mSegment_SpectralClusters;

	int numSamples = inGraph.rows();
	int numNonZeros = inGraph.nonZeros();

	// compute Laplacian (affinity matrix)
	cout << "Computing Laplacian" << endl;

	Eigen::SparseMatrix<double> affMat = inGraph;
	vector<double> rowSum(numSamples, 0.0);
	for (int k = 0; k < affMat.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(affMat, k); it; ++it) {
			rowSum[it.row()] += it.value();
		}
	}
	for (auto &value : rowSum) {
		value = 1.0 / (sqrt(value) + 1e-30);
	}
	for (int k = 0; k < affMat.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(affMat, k); it; ++it) {
			it.valueRef() *= rowSum[it.row()] * rowSum[it.col()];
		}
	}

	// compute eigenvalues & eigenvectors
	cout << "Solving eigenvalues" << endl;

	vector<ARPACKHelper::TTriplet> triplets;
	triplets.reserve(affMat.nonZeros());
	for (int k = 0; k < affMat.outerSize(); k++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(affMat, k); it; ++it) {
			triplets.push_back(ARPACKHelper::TTriplet(it.row(), it.col(), it.value()));
		}
	}
	vector<double> eigenValues;
	vector<vector<double>> eigenVectors;
	if (!ARPACKHelper::compute(numSamples, numClusters, triplets, eigenValues, eigenVectors, eps)) return false;

	// k-means clustering
	cout << "Applying K-means clustering..." << endl;

	Eigen::MatrixXd specMat(numSamples, numClusters);
#pragma omp parallel for
	for (int j = 0; j < numClusters; j++) {
		specMat.col(j) = Eigen::VectorXd::Map(eigenVectors[j].data(), numSamples);
	}
	specMat.rowwise().normalize();
	if (!specMat.allFinite()) {
		cout << "Error: normalization error in spectral clustering" << endl;
		return false;
	}
	Eigen::VectorXi clusterIndices;
	if (!ClusterKMeans::cluster(numClusters, specMat, clusterIndices)) return false;

	// extract connected components
	cout << "Extracting connected components..." << endl;
	if (!connect(inGraph, clusterIndices)) return false;

	outCluster.swap(clusterIndices);

	cout << "Spectral Clustering: Done." << endl;

	return true;
}

bool ClusterSparseSpectral::connect(Eigen::SparseMatrix<double> &inGraph, Eigen::VectorXi &inoutCluster) {

	// build neighbor sets

	int numSamples = inGraph.rows();
	vector<set<int>> neighborSet(numSamples);
	for (int k = 0; k < inGraph.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(inGraph, k); it; ++it) {
			int r = it.row();
			int c = it.col();
			neighborSet[r].insert(c);
			neighborSet[c].insert(r);
		}
	}

	// find connected components

	Eigen::VectorXi newCluster(numSamples);
	newCluster.fill(-1);
	int clusterCount = 0;
	int minSize = INT_MAX;
	for (int sampleID = 0; sampleID < numSamples; sampleID++) {

		if (newCluster[sampleID] >= 0) continue;
		newCluster[sampleID] = clusterCount;

		// BFS
		vector<int> queue(1, sampleID);
		int head = 0;
		while (head < (int)queue.size()) {
			int point = queue[head];
			for (int neighbor : neighborSet[point]) {
				if (newCluster[neighbor] < 0 && inoutCluster[point] == inoutCluster[neighbor]) {
					newCluster[neighbor] = clusterCount;
					queue.push_back(neighbor);
				}
			}
			head++;
		}
		clusterCount++;
		if ((int)queue.size() < minSize) minSize = (int)queue.size();
	}

	cout << "Number of connected components: " << clusterCount << endl;
	cout << "Min size: " << minSize << endl;

	inoutCluster.swap(newCluster);

	return true;
}