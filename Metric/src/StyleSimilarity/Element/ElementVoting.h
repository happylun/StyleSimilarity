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

#include "Data/StyleSimilarityTypes.h"
#include "Data/StyleSimilarityData.h"

using namespace std;

namespace StyleSimilarity {

	class ElementVoting {

	private:

		struct TTransformVote {
			int patchSourceID;
			int patchTargetID;
			double weight;
			Eigen::Affine3d transformation;
			TTransformVote() : weight(0) {}
			inline bool operator==(const TTransformVote &other) {
				return weight == other.weight;
			} // for remove
			inline bool operator<(const TTransformVote &other) {
				return weight > other.weight;
			} // for sorting
		};

	public:

		ElementVoting(StyleSimilarityData *data);
		~ElementVoting();

	public:

		bool saveVotes(string fileName);
		bool loadVotes(string fileName);

		bool saveVoteDistance(string fileName);
		bool loadVoteDistance(string fileName);

		bool saveGlobalAlignment(string fileName);
		bool loadGlobalAlignment(string fileName);

		bool saveVoteModes(string fileName);
		bool loadVoteModes(string fileName);

		bool saveModes(string fileName);
		bool loadModes(string fileName);

	public:

		bool computeVotes();
		bool computeVoteDistances();
		bool computeGlobalAlignment();
		bool clusterVotes(double sigma = 1.0);
		bool adjustModes();

		bool visualizeModes(string fileName);
		bool exportTSpaceData(string voteFileName, string modeFileName);

	private:

		bool checkPatchPairs(
			double sourceVolume,
			double targetVolume,
			Eigen::Vector3d &sourceVariance,
			Eigen::Vector3d &targetVariance);
		bool computePatchIntrinsics(
			Eigen::Matrix3Xd &inPatch,
			double &inRadius,
			double &outVolume,
			Eigen::Vector3d &outVariance);
		bool computePatchWeight(
			Eigen::Matrix3Xd &sourceP,
			Eigen::Matrix3Xd &sourceN,
			Eigen::Matrix3Xd &targetP,
			Eigen::Matrix3Xd &targetN,
			double radius,
			double &weight);

		void error(string s);

	private:

		StyleSimilarityData *mpData;
		vector<TTransformVote> mTransformationVotes;
		Eigen::MatrixXd mTransformationVoteDistance; // # of votes X dimD
		Eigen::Affine3d mGlobalTransformation;
		Eigen::VectorXd mGlobalDistance;
		vector<int> mTransformationModeFromVote; // vote ID : # of modes
		vector<int> mTransformationVoteFromMode; // mode ID or -1 : # of votes

		Eigen::MatrixXd mTSpaceVotes; // # of votes X 10 (9-dimensional vote + weight)
		Eigen::MatrixXd mTSpaceModes; // # of modes X 9
	};
}