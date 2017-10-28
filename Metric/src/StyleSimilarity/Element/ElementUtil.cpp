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

#include "ElementUtil.h"

#include <fstream>

#include <Library/CMLHelper.h>

#include "Data/StyleSimilarityConfig.h"

using namespace StyleSimilarity;

bool ElementUtil::convertAffineToTRS(Eigen::Affine3d &inAffine, vector<double> &outTRS) {

	Eigen::Vector3d vecT = inAffine.translation();

	Eigen::Matrix3d matLinear = inAffine.linear();
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(matLinear, Eigen::ComputeFullU | Eigen::ComputeFullV);
	double x = (svd.matrixU() * svd.matrixV().adjoint()).determinant();
	Eigen::Vector3d vecS = svd.singularValues();
	vecS.coeffRef(0) *= x;

	Eigen::Matrix3d matU = svd.matrixU();
	matU.col(0) /= x;
	Eigen::Matrix3d matR = matU * svd.matrixV().adjoint();
	Eigen::Quaterniond vecR(matR);
	if (vecR.w() < 0) {
		vecR.w() = -vecR.w();
		vecR.x() = -vecR.x();
		vecR.y() = -vecR.y();
		vecR.z() = -vecR.z();
	}
	vecR.normalize();

	outTRS.clear();
	outTRS.push_back(vecT.x());
	outTRS.push_back(vecT.y());
	outTRS.push_back(vecT.z());
	outTRS.push_back(vecR.w());
	outTRS.push_back(vecR.x());
	outTRS.push_back(vecR.y());
	outTRS.push_back(vecR.z());
	outTRS.push_back(vecS.x());
	outTRS.push_back(vecS.y());
	outTRS.push_back(vecS.z());

	return true;
}
/*
bool ElementUtil::convertTRSToAffine(vector<double> &inTRS, Eigen::Affine3d &outAffine) {

	if ((int)inTRS.size() != 10) {
		cout << "Error: incorrect size of TRS vector (should be 10)" << endl;
		return false;
	}

	Eigen::Vector3d tVec(inTRS[0], inTRS[1], inTRS[2]);
	Eigen::Quaterniond rVec(inTRS[3], inTRS[4], inTRS[5], inTRS[6]);
	Eigen::Vector3d sVec(inTRS[7], inTRS[8], inTRS[9]);
	outAffine.setIdentity();
	outAffine.prescale(sVec);
	outAffine.prerotate(rVec.normalized());
	outAffine.pretranslate(tVec);

	return true;
}
*/

bool ElementUtil::saveVectorASCII(string fileName, Eigen::VectorXd &inVector) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << inVector << endl;
	file.close();

	return true;
}

bool ElementUtil::loadVectorASCII(string fileName, Eigen::VectorXd &outVector) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	vector<double> data;
	while (!file.eof()) {
		double value;
		file >> value;
		if (file.fail()) break;
		data.push_back(value);
	}
	file.close();

	outVector.resize(data.size());
	for (int pos = 0; pos < (int)data.size(); pos++) {
		outVector(pos) = data[pos];
	}

	return true;
}

bool ElementUtil::saveRowVectorASCII(string fileName, Eigen::RowVectorXd &inVector) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << inVector << endl;
	file.close();

	return true;
}

bool ElementUtil::loadRowVectorASCII(string fileName, Eigen::RowVectorXd &outVector) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	vector<double> data;
	while (!file.eof()) {
		double value;
		file >> value;
		if (file.fail()) break;
		data.push_back(value);
	}
	file.close();

	outVector.resize(data.size());
	for (int pos = 0; pos < (int)data.size(); pos++) {
		outVector(pos) = data[pos];
	}

	return true;
}

bool ElementUtil::saveMatrixASCII(string fileName, Eigen::MatrixXd &inMatrix) {

	ofstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	file << inMatrix << endl;
	file.close();

	return true;
}

bool ElementUtil::loadMatrixASCII(string fileName, Eigen::MatrixXd &outMatrix) {

	ifstream file(fileName);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int dim = -1;
	vector<vector<double>> data;
	while (!file.eof()) {
		string line;
		getline(file, line);
		if (!file.good()) break;
		stringstream ss(line);
		vector<double> row;
		while (!ss.eof()) {
			double value;
			ss >> value;
			if (ss.fail()) break;
			row.push_back(value);
		}
		if (row.empty()) break;
		if (dim == -1) dim = (int)row.size();
		if (dim != (int)row.size()) {
			cout << "Error: input matrix has inconsistent dimensions" << endl;
			return false;
		}
		data.push_back(row);
	}
	file.close();

	outMatrix.resize(data.size(), dim);
	for (int row = 0; row < (int)data.size(); row++) {
		for (int col = 0; col < dim; col++) {
			outMatrix(row, col) = data[row][col];
		}
	}

	return true;
}


bool ElementUtil::saveVectorBinary(string fileName, Eigen::VectorXd &inVector) {

	ofstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	int length = (int)inVector.size();
	file.write((const char*)(&length), sizeof(length));
	for (int l = 0; l < length; l++) {
		double value = inVector(l);
		file.write((const char*)(&value), sizeof(value));
	}
	file.close();

	return true;
}

bool ElementUtil::loadVectorBinary(string fileName, Eigen::VectorXd &outVector) {

	ifstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int length;
	file.read((char*)(&length), sizeof(length));
	outVector.resize(length);
	for (int l = 0; l < length; l++) {
		double value;
		file.read((char*)(&value), sizeof(value));
		outVector(l) = value;
	}
	file.close();

	return true;
}

bool ElementUtil::saveRowVectorBinary(string fileName, Eigen::RowVectorXd &inVector) {

	ofstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	int length = (int)inVector.size();
	file.write((const char*)(&length), sizeof(length));
	for (int l = 0; l < length; l++) {
		double value = inVector(l);
		file.write((const char*)(&value), sizeof(value));
	}
	file.close();

	return true;
}

bool ElementUtil::loadRowVectorBinary(string fileName, Eigen::RowVectorXd &outVector) {

	ifstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int length;
	file.read((char*)(&length), sizeof(length));
	outVector.resize(length);
	for (int l = 0; l < length; l++) {
		double value;
		file.read((char*)(&value), sizeof(value));
		outVector(l) = value;
	}
	file.close();

	return true;
}

bool ElementUtil::saveMatrixBinary(string fileName, Eigen::MatrixXd &inMatrix) {

	ofstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}
	int rows = (int)inMatrix.rows();
	int cols = (int)inMatrix.cols();
	file.write((const char*)(&rows), sizeof(rows));
	file.write((const char*)(&cols), sizeof(cols));
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double value = inMatrix(r, c);
			file.write((const char*)(&value), sizeof(value));
		}
	}
	file.close();

	return true;
}

bool ElementUtil::loadMatrixBinary(string fileName, Eigen::MatrixXd &outMatrix) {

	ifstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}
	int rows, cols;
	file.read((char*)(&rows), sizeof(rows));
	file.read((char*)(&cols), sizeof(cols));
	outMatrix.resize(rows, cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double value;
			file.read((char*)(&value), sizeof(value));
			outMatrix(r, c) = value;
		}
	}
	file.close();

	return true;
}

bool ElementUtil::saveCellArrayBinary(string fileName, vector<int> &inArray) {

	ofstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}

	int length = (int)inArray.size();
	file.write((const char*)(&length), sizeof(length));
	for (int k = 0; k < length; k++) {
		int value = inArray[k];
		file.write((const char*)(&value), sizeof(value));
	}

	file.close();

	return true;
}

bool ElementUtil::loadCellArrayBinary(string fileName, vector<int> &outArray) {

	ifstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	int length;
	file.read((char*)(&length), sizeof(length));
	outArray.resize(length);
	for (int k = 0; k < length; k++) {
		int value;
		file.read((char*)(&value), sizeof(value));
		outArray[k] = value;
	}

	file.close();

	return true;
}

bool ElementUtil::saveCellArraysBinary(string fileName, vector<vector<int>> &inArrays) {

	ofstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot write to file " << fileName << endl;
		return false;
	}

	int rows = (int)inArrays.size();
	file.write((const char*)(&rows), sizeof(rows));
	for (int r = 0; r < rows; r++) {
		auto &row = inArrays[r];
		int length = (int)row.size();
		file.write((const char*)(&length), sizeof(length));
		for (int k = 0; k < length; k++) {
			int value = row[k];
			file.write((const char*)(&value), sizeof(value));
		}
	}

	file.close();

	return true;
}

bool ElementUtil::loadCellArraysBinary(string fileName, vector<vector<int>> &outArrays) {

	ifstream file(fileName, ios::binary);
	if (!file.is_open()) {
		cout << "Error: cannot open file " << fileName << endl;
		return false;
	}

	int rows;
	file.read((char*)(&rows), sizeof(rows));
	outArrays.resize(rows);
	for (int r = 0; r < rows; r++) {
		auto &row = outArrays[r];
		int length;
		file.read((char*)(&length), sizeof(length));
		row.resize(length);
		for (int k = 0; k < length; k++) {
			int value;
			file.read((char*)(&value), sizeof(value));
			row[k] = value;
		}
	}

	file.close();

	return true;
}