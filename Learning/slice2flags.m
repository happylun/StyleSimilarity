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

function flags = slice2flags( slice, len )

    flags = zeros(max(1,size(slice,1)), len);
    for k = 1:size(slice,1)
        flags(k,slice{k}) = flags(k,slice{k}) + 1;
    end
end

