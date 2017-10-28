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

#include "Timer.h"

#include <sstream>

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>

__int64 Timer::frequency = 0;
__int64 Timer::counter = 0;

void Timer::tic() {
	if (frequency == 0) QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
	QueryPerformanceCounter((LARGE_INTEGER*)&counter);
}

int Timer::toc() {
	__int64 now;
	QueryPerformanceCounter((LARGE_INTEGER*)&now);
	return (int)(1000*(now-counter)/frequency);
}

string Timer::toString(int time) {
	int ms = time % 1000; time /= 1000;
	int s = time % 60; time /= 60;
	int m = time % 60; time /= 60;
	int h = time;
	stringstream ss;
	ss << h << " H " << m << " M " << s << " S " << ms << endl;
	return ss.str();
}