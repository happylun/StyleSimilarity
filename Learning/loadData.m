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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% data format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{

data.ti: N*3 matrix
    triplet index: each row is a triplet of ID of 3 shapes (ID starts from 1)
data.ta: N*1 vector
    tripelt answer: row n is the majority answer (1~4) of triplet n
data.tw: N*4 vector
    triplet weight: row n is the weight (confidence) of triplet n (4 answers)
data.s: M*1 cell
    saliency: row m contains point saliencies for shape m
    within each cell: K*D matrix
        row k is the D-dimension saliency feature vector for point k
data.pd|sd|as|at|us|ut|ms|mt: N*2 cell
    row n contains info for triplet n
    first column contains info between shape A and shape B
    second column contains info between shape A and shape C
data.pd cell: S*D matrix
    patch distance: row s is the D-dimension distance feature vector for patch-pair s
data.sd cell: 1*D matrix
    D-dimension distance feature vector for entire shapes
data.as cell: S*K matrix
    entry (s, k) is 1 if point k belongs to element s on shape A
data.at cell: S*K matrix
    entry (s, k) is 1 if point k belongs to element s on shape B(C)
data.us cell: 1*K matrix
    column k is 1 if point k does not belong to any element on shape A
data.ut cell: 1*K matrix
    column k is 1 if point k does not belong to any element on shape B(C)
data.ms cell: 1*K matrix
    column k is number of elements point k belongs to on shape A
data.mt cell: 1*K matrix
    column k is number of elements point k belongs to on shape B(C)

%}

clear all;
close all;
clc;
inputFolder = '../../Data/demo/';
dataFolder = inputFolder;

%%%%%%%%%%%%%%%%%%%%%%%% data-specific parameter %%%%%%%%%%%%%%%%%%%%%%%%

dataAffix = '-2'; % sigma
subSampleStride = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

fprintf('Loading triplets\n');
responseFolder = strcat(dataFolder, '/response/');
data.ti = load(strcat(responseFolder, 'tripletIndex.txt'));
data.ta = load(strcat(responseFolder, 'tripletAnswer.txt'));
data.tw = load(strcat(responseFolder, 'tripletWeight.txt'));
numTriplets = size(data.ti, 1);
numShapes = max(max(data.ti));

fprintf('Loading saliency\n');
data.s = cell(numShapes, 1); % point saliency: # of points * dimS
shapeNumSamples = zeros(numShapes, 1);
for k = 1:numShapes
    data.s{k} = loadMatrix(sprintf('%s/saliency/%d.txt', inputFolder, k));
    shapeNumSamples(k) = size(data.s{k}, 1);
    if subSampleStride > 2
        data.s{k} = data.s{k}(1:subSampleStride:end, :);
    end
    if mod(k,10)==0, fprintf('.'), end
end
fprintf('\n');

pairList = {'AB', 'AC'};

fprintf('Loading distance\n');
data.pd = cell(numTriplets, 2); % patch distance: # of elements * dimPD
data.sd = cell(numTriplets, 2); % shape distance: # of elements * dimSD
for k = 1:numTriplets
    for j = 1:2
        data.pd{k, j} = loadMatrix(sprintf('%s/triplet/%d/pd%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
        data.sd{k, j} = loadVector(sprintf('%s/triplet/%d/sd%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
    end
    if mod(k,10)==0, fprintf('.'), end
end
fprintf('\n');

fprintf('Loading index\n');
data.as = cell(numTriplets, 2); % element source flags: # elements * # of points
data.at = cell(numTriplets, 2); % element target flags: # elements * # of points
data.us = cell(numTriplets, 2); % unmatch source flags: 1 * # of points
data.ut = cell(numTriplets, 2); % unmatch target flags: 1 * # of points
data.ms = cell(numTriplets, 2); % number of elements each point belongs to: 1 * # of points
data.mt = cell(numTriplets, 2); % number of elements each point belongs to: 1 * # of points
for k = 1:numTriplets
    for j = 1:2
        ns = shapeNumSamples(data.ti(k,1));
        nt = shapeNumSamples(data.ti(k,j+1));
        as = loadCellArray(sprintf('%s/triplet/%d/as%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
        at = loadCellArray(sprintf('%s/triplet/%d/at%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
        us = loadArray(sprintf('%s/triplet/%d/us%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
        ut = loadArray(sprintf('%s/triplet/%d/ut%s%s.txt', inputFolder, k, pairList{j}, dataAffix));
        data.as{k, j} = slice2flags(as, ns);
        data.at{k, j} = slice2flags(at, nt);
        data.us{k, j} = slice2flags({us}, ns);
        data.ut{k, j} = slice2flags({ut}, nt);
        if subSampleStride > 2
            data.as{k,j} = data.as{k,j}(:, 1:subSampleStride:end);
            data.at{k,j} = data.at{k,j}(:, 1:subSampleStride:end);
            data.us{k,j} = data.us{k,j}(:, 1:subSampleStride:end);
            data.ut{k,j} = data.ut{k,j}(:, 1:subSampleStride:end);
        end
        data.ms{k,j} = sum(data.as{k,j},1);
        data.mt{k,j} = sum(data.at{k,j},1);
    end
    if mod(k,10)==0, fprintf('.'), end
end
fprintf('\n');

reliableSlice = find(sum(data.tw,2) > 0);
fprintf('Number of informative queries: %d\n', length(reliableSlice));

dimPD = size(data.pd{1,1},2);
dimSD = size(data.sd{1,1},2);
dimS = size(data.s{1,1},2);

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%% normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

maxPercentile = 90;

allPD = cell2mat(reshape(data.pd(reliableSlice,:),[],1));
allPDScale = prctile(allPD, maxPercentile, 1); % 1 * dimPD
allPDScale(allPDScale==0) = 1;

allSD = cell2mat(reshape(data.sd(reliableSlice,:),[],1));
allSDScale = prctile(allSD, maxPercentile, 1); % 1 * dimSD
allSDScale(allSDScale==0) = 1;

for  k =1:numTriplets
    for j = 1:2
        if size(data.pd{k,j},1) == 0
            data.pd{k,j} = ones(1, dimPD); % no element
        else
            data.pd{k,j} = bsxfun(@rdivide, data.pd{k,j}, allPDScale);
            data.pd{k,j} = min(data.pd{k,j}, 1.0);
        end
        
        data.sd{k,j} = data.sd{k,j} ./ allSDScale;
        data.sd{k,j} = min(data.sd{k,j}, 1.0);
    end
end

allS = cell2mat(data.s);
allSScale = prctile(allS, maxPercentile, 1); % 1 * dimS
allSScale(allSScale==0) = 1;

for k = 1:numShapes
    data.s{k} = bsxfun(@rdivide, data.s{k}, allSScale);
    data.s{k} = min(data.s{k}, 1.0);
end

toc;

