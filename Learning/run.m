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

clc;

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   cutlery  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-cutlery';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   building  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-building';
dataAffix = '-4'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   coffee set  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-coffeeset';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   column  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-column';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   dish  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-dish';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   furniture  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-furniture';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn

clear all;
fprintf('\n\n\n\n\n|||||||||||||||||||||||||||||||||   lamp  |||||||||||||||||||||||||||||||||||\n');
dataFolder = 'data/big-lamp';
dataAffix = '-2'; % sigma
loadData
learn
%anotherlearn