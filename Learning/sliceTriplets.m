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

function  slice = sliceTriplets( n, k )

slice = cell(k, 1);
p = 1;
for i = 1:k
    q = floor(i*n/k);
    slice{i} = p:q;
    p = q+1;
end

end

