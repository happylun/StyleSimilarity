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

function C = loadCellArray(name)

file = fopen(name);
rows = fread(file, [1,1], 'int');
C = cell(rows, 1);
for r = 1:rows
    len = fread(file, [1,1], 'int');
    C{r} = fread(file, len, 'int');
    C{r} = C{r}+1; % convert 0-indexed to 1-indexed
end
fclose(file);
