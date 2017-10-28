%=========================================================================
%
% This file is part of the Style Similarity project.
%
% Copyright (c) 2015 - Zhaoliang Lun (author of the code) / UMass-Amherst
%
% This is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This software is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this software.  If not, see <http://www.gnu.org/licenses/>.
%
%=========================================================================

function M = loadMatrix(name)

file = fopen(name);
size = fread(file, [2,1], 'int');
M = fread(file, [size(2), size(1)], 'double')';
fclose(file);
if nnz(isnan(M))>0
    [r,c] = find(isnan(M));
    fprintf('\nWarning: NaN in file "%s"\n', name);
    fprintf('row %d, col %d\n', [r'; c']);
end
if nnz(isinf(M))>0
    [r,c] = find(isinf(M));
    fprintf('\nWarning: Inf in file "%s"\n', name);
    fprintf('row %d, col %d\n', [r'; c']);
end
M(isnan(M)) = 0;
M(isinf(M)) = 0;