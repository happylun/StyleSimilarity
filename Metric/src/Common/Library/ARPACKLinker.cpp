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

#include "ARPACKHelper.h"

#include <cmath>
#include <iostream>

using namespace std;

extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which,
	int *nev, double *tol, double *resid, int *ncv,
	double *v, int *ldv, int *iparam, int *ipntr,
	double *workd, double *workl, int *lworkl,
	int *info);

extern "C" void dseupd_(int *rvec, char *All, int *select, double *d,
	double *z, int *ldz, double *sigma,
	char *bmat, int *n, char *which, int *nev,
	double *tol, double *resid, int *ncv, double *v,
	int *ldv, int *iparam, int *ipntr, double *workd,
	double *workl, int *lworkl, int *ierr);

bool ARPACKHelper::compute(
	int inNumDim, int inNumEV,
	vector<TTriplet> &inMatrix,
	vector<double> &outValues,
	vector<vector<double>> &outVectors,
	double eps)
{
	int ido = 0;
	char bmat[2] = "I"; // A * v = lambda * I * v
	char which[3] = "LM"; // eigen values are sorted by largest magnitude
	double tol = eps; // 0: use machine eps
	double *resid = new double[inNumDim];
	int ncv = 2 * inNumEV; // largest number of basis vectors
	if (ncv>inNumDim) ncv = inNumDim;
	int ldv = inNumDim;
	double *v = new double[ldv*ncv];
	int *iparam = new int[11];
	iparam[0] = 1;
	iparam[2] = 3 * inNumDim; // max number of iterations
	iparam[6] = 1;
	int *ipntr = new int[11];
	double *workd = new double[3 * inNumDim];
	double *workl = new double[ncv*(ncv + 8)];
	int lworkl = ncv*(ncv + 8);
	int info = 0;
	int rvec = 1; // calculate eigenvectors
	int *select = new int[ncv];
	double *d = new double[2 * ncv];
	double sigma;
	int ierr;

	do {
		dsaupd_(&ido, bmat, &inNumDim, which, &inNumEV, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &info);

		if ((ido == 1) || (ido == -1)) {
			if (!product(inNumDim, inMatrix, workd + ipntr[0] - 1, workd + ipntr[1] - 1)) return false;
		}
	} while ((ido == 1) || (ido == -1));

	if (info<0) {
		cout << "Error with dsaupd, info = " << info << "\n";
		cout << "Check documentation in dsaupd\n\n";
	} else {
		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
			&inNumDim, which, &inNumEV, &tol, resid, &ncv, v, &ldv,
			iparam, ipntr, workd, workl, &lworkl, &ierr);

		if (ierr != 0) {
			cout << "Error with dseupd, info = " << ierr << "\n";
			cout << "Check the documentation of dseupd.\n\n";
			return false;
		} else if (info == 1) {
			cout << "Maximum number of iterations reached.\n\n";
			return false;
		} else if (info == 3) {
			cout << "No shifts could be applied during implicit\n";
			cout << "Arnoldi update, try increasing NCV.\n\n";
			return false;
		}

		outValues.resize(inNumEV);
		outVectors.resize(inNumEV);
		for (int i = 0; i < inNumEV; i++) {
			outValues[i] = d[i];
			outVectors[i].assign(v + i*inNumDim, v + (i + 1)*inNumDim);
		}
	}

	delete[] resid;
	delete[] v;
	delete[] iparam;
	delete[] ipntr;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] d;

	return true;
}

bool ARPACKHelper::product(int n, vector<TTriplet> &mat, double *in, double*out) {

	// out = mat * in
	for (int i = 0; i < n; i++) out[i] = 0;
	for (auto &it : mat) out[it.row] += in[it.col] * it.value;

	return true;
}