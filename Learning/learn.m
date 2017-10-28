%=========================================================================
%
% This file is part of the Style Similarity project.
%
% Copyright (c) 2015 - Zhaoliang Lun, Evangelos Kalogerakis  / UMass-Amherst
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

%%%%%%%%%%%%%%%%%%%%%%%%%%% global parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

OBJECTIVE_FUNCTION = 'MAP'; % {'LMNN', 'MAP'}
numFolds = 2; % 2 for demo; 10 for real data

param.useSigmoidsDistance = false;
param.useSigmoidsSaliency = true;
param.separateTrainTest = true;
param.noCrossValidation = false;
param.skipMatchedTerm = false;
param.skipUnmatchedTerm = false;
param.skipSaliencyTerm = false;
param.skipElementDistance = false;
param.skipGlobalDistance = true;
alternatePrefix = '';

clear('fixedLambdaD', 'var');
clear('fixedLambdaS', 'var');
clear('fixedLambdaU', 'var');

fixedLambdaD = 0.001;
fixedLambdaS = 0.01;
fixedLambdaU = 0.1;

gridSearch = ~exist('fixedLambdaD', 'var') || ~exist('fixedLambdaS', 'var') || ~exist('fixedLambdaU', 'var');

spdSlice = [1:2 18:73]; % slice for simple patch distance

%%%%%%%%%%%%%%%%%%%%%%%%%%% learning %%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Learning...\n');
learnTime = tic;

lambdaLog = [];
lambdaRange = [1e-3 1e-2 1e-1]; % used for grid search
if ~exist('fixedLambdaD', 'var')
    allLambdaD = lambdaRange;
else
    allLambdaD = fixedLambdaD;
end
if ~exist('fixedLambdaS', 'var')
    allLambdaS = lambdaRange;
else
    allLambdaS = fixedLambdaS;
end
if ~exist('fixedLambdaU', 'var')
    allLambdaU = lambdaRange;
else
    allLambdaU = fixedLambdaU;
end

outFailCases = [];
outSuccCases = [];
outFailDistances = [];
outSuccDistances = [];

for lambdaD = allLambdaD
for lambdaS = allLambdaS
for lambdaU = allLambdaU
fprintf('lambdaD = %f, lambdaS = %f, lambdaU = %f\n', lambdaD, lambdaS, lambdaU);
lambdaAll = [lambdaD lambdaS lambdaU];

timeAffix = datestr(now, '--mm-dd-HHMMSS');

testDistances = zeros(numTriplets, 2);
foldIdx = randfold(length(reliableSlice), numFolds);
if param.noCrossValidation
    numFolds = 0;
end
for fold = 1:numFolds+1
    
    fprintf('---------------------------\n');
    
    if fold > numFolds
        % train on all reliable triplets
        testSet = setdiff(1:numTriplets, reliableSlice);
    else
        % slices for training/testing data
        testSet = reliableSlice(foldIdx==fold);
        fprintf('Test set: ');
        fprintf('%d ', testSet);
        fprintf('\n');
    end
    trainSet = setdiff(reliableSlice, testSet);
    if param.separateTrainTest && fold <= numFolds
        testTriplets = data.ti(testSet,:);
        testPairs = unique([testTriplets(:,[1 2]); testTriplets(:,[2 1]); ...
            testTriplets(:,[1 3]); testTriplets(:,[3 1])], 'rows');
        allPairs = [data.ti(:,[1 2]); data.ti(:,[1 3])];
        trainSet = intersect(trainSet, find(~any(reshape(ismember(allPairs, testPairs, 'rows'), [], 2), 2))');
        if ~isempty(intersect(trainSet, testSet))
            error('Error in separating training set and testing set');
        end
    end
    
    fprintf('Train set: %d, Test set: %d\n', length(trainSet), length(testSet));
    
    % initial weights
    if ~param.useSigmoidsDistance
        Wd = ones(dimSD, 1) / dimSD; % distance
        Wdb = 1e-30; % bias weight for distance (unused)
    else
        Wd = rand(dimSD, 1) / dimSD; % distance
        Wdb = -1; % bias weight for distance        
    end
    if ~param.useSigmoidsSaliency
        Ws = ones(dimS, 1) / dimS; % saliency
        Wsb = 1e-30; % bias weight for point saliency (unused)
    else
        Ws = randn(dimS, 1) / dimS; % saliency
        Wsb = 1e-30; % bias weight for point saliency
    end
    Wu = .001; % unmatched term (small enough to be strictly > 0)
    
    if param.skipMatchedTerm
        Wd = zeros(dimSD, 1);
        Wdb = 0;
    end
    if param.skipUnmatchedTerm
        Wu = 0;
    end
    if param.skipSaliencyTerm
        Ws = zeros(dimS, 1);
        Wsb = 0;
    end
    
    weights = [Wd; Ws; Wu; Wdb; Wsb];
    lambda = [ones(dimSD,1)*lambdaAll(1); ones(dimS,1)*lambdaAll(2); lambdaAll(3); lambdaAll(1); lambdaAll(2)];
    
    if strcmp(OBJECTIVE_FUNCTION, 'LMNN')
        objFunc = @LMNNObjective;
    elseif strcmp(OBJECTIVE_FUNCTION, 'MAP')
        objFunc = @MAPObjective;
    end
    
    lb = 1e-30 * ones(size(weights));
    if param.useSigmoidsDistance        
        lb(dimSD+dimS+2) = -Inf; % Wdb
    end
    if param.useSigmoidsSaliency
        lb(dimSD+1:dimSD+dimS) = -Inf(dimS,1); % Ws
        lb(dimSD+dimS+3) = -Inf; % Wsb
    end
    
    % optimization with Matlab's fmincon
    options = optimset(            ...
        'Algorithm', 'sqp',        ...
        'DerivativeCheck', 'off',  ...
        'Display', 'final',        ...
        'TolFun', 1e-4,            ...
        'MaxIter', 100,            ...
        'GradObj', 'on');
    
    fminconTime = tic;
    [weights, objValue] = fmincon(                         ...
        @(x) objFunc(x, data, trainSet, lambda, param),    ...
        weights, [], [], [], [], lb, [], [], options);
    toc(fminconTime);
    
    Wd = weights(1:dimSD);
    Ws = weights(dimSD+1:dimSD+dimS);
    Wu = weights(dimSD+dimS+1);
    Wdb = weights(dimSD+dimS+2);
    Wsb = weights(dimSD+dimS+3);
    
    % {
    % display weights
    fprintf('\nWd: ');
    fprintf('%g ', Wd);
    fprintf('\nWs: ');
    fprintf('%g ', Ws);
    fprintf('\nWu: %g', Wu);
    fprintf('\nWdb: %g\nWsb: %g', Wdb, Wsb);
    fprintf('\nObjective: %g\n', objValue);
    %}
    
    [distance, ~] = computeDistance(data, weights, 1:numTriplets, param);
    
    if ~param.noCrossValidation
		testDistances(testSet, :) = distance(testSet, :);
    end
    
    [~, predictedAnswer] = min(distance, [], 2);
    predictedResult = double(predictedAnswer == data.ta);

    % check training data
    resTr = predictedResult(trainSet);
    aTr = nnz(resTr)/length(resTr);
    fprintf('Training data correct rate: %d / %d = %f\n', nnz(resTr), length(resTr), aTr);

    if fold <= numFolds
        % check testing data
        resTe = predictedResult(testSet);
        aTe = nnz(resTe)/length(resTe);
        fprintf('Testing data correct rate: %d / %d = %f\n', nnz(resTe), length(resTe), aTe);
    end
    
    if fold > numFolds
        
        weightFolder = strcat(dataFolder, '/weight/');
        if ~exist(weightFolder, 'dir')
            mkdir(weightFolder);
        end
        
        % export all weights
        wFile = fopen(strcat(weightFolder, alternatePrefix, 'data-all-weights', timeAffix, '.txt'), 'w');
        fprintf(wFile, '\nWd: ');
        fprintf(wFile, '%g ', Wd);
        fprintf(wFile, '\nWs: ');
        fprintf(wFile, '%g ', Ws);
        fprintf(wFile, '\nWu: %g', Wu);
        fprintf(wFile, '\nWdb: %g\nWsb: %g', Wdb, Wsb);
        fprintf(wFile, '\nObjective: %g\n', objValue);
        fprintf(wFile, 'Lambda D/S/U: %f\n', [lambdaD lambdaS lambdaU]);
        fclose(wFile);
        
        % export normalized weights
        wFile = fopen(strcat(weightFolder, alternatePrefix, 'data-normalized-weights', timeAffix, '.txt'), 'w');
        denom = sum(Wd) + Wu;
        fprintf(wFile, 'Wd: ');
        fprintf(wFile, '%g ', Wd/denom);
        fprintf(wFile, '\nWs: ');
        fprintf(wFile, '%g ', abs(Ws)/sum(abs(Ws)));
        fprintf(wFile, '\nWu: %g\n', Wu/denom);
        fclose(wFile);
        
        % export simple patch distance weights
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-weight-SPD', timeAffix, '.txt'), Wd(spdSlice));
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-scale-SPD', timeAffix, '.txt'), allPDScale(spdSlice)');
        
        % export full patch distance weights
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-weight-FPD', timeAffix, '.txt'), Wd);
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-scale-FPD', timeAffix, '.txt'), allPDScale');
        
        % export full saliency weights
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-weight-FS', timeAffix, '.txt'), [Ws;Wsb]);
        dlmwrite(strcat(weightFolder, alternatePrefix, 'data-scale-FS', timeAffix, '.txt'), allSScale');
    end
end

[~, predictedAnswer] = min(testDistances, [], 2);
predictedResult = double(predictedAnswer == data.ta);
failCases = intersect(reliableSlice, find(~predictedResult));
succCases = setdiff(reliableSlice, failCases);
accuracy = length(succCases)/length(reliableSlice);

fprintf('===========================\n');
fprintf('lambdaD = %f, lambdaS = %f, lambdaU = %f\n', lambdaD, lambdaS, lambdaU);
fprintf('Cross validation accuracy: %d / %d = %f\n', length(succCases), length(reliableSlice), accuracy);

% {
if ~param.noCrossValidation
    
    caseFolder = strcat(dataFolder, '/cases');
    if ~exist(caseFolder, 'dir')
        mkdir(caseFolder);
    end
    
    paramAffix = '-000000000';
    if param.useSigmoidsDistance; paramAffix(2) = '1'; end
    if param.useSigmoidsSaliency; paramAffix(3) = '1'; end
    if param.separateTrainTest; paramAffix(4) = '1'; end
    if param.noCrossValidation; paramAffix(5) = '1'; end
    if param.skipMatchedTerm; paramAffix(6) = '1'; end
    if param.skipUnmatchedTerm; paramAffix(7) = '1'; end
    if param.skipSaliencyTerm; paramAffix(8) = '1'; end
    if param.skipElementDistance; paramAffix(9) = '1'; end
    if param.skipGlobalDistance; paramAffix(10) = '1'; end
    
    dlmwrite(strcat(caseFolder, '/', alternatePrefix, 'failCases', timeAffix, paramAffix, '.txt'), failCases, ' ');
    dlmwrite(strcat(caseFolder, '/', alternatePrefix, 'succCases', timeAffix, paramAffix, '.txt'), succCases, ' ');
    dlmwrite(strcat(caseFolder, '/', alternatePrefix, 'distances', timeAffix, paramAffix, '.txt'), testDistances, ' ');
end
%}

if gridSearch
    lambdaLog = [lambdaLog; lambdaD lambdaS lambdaU accuracy];
    dlmwrite(strcat(dataFolder, '/lambda-log', timeAffix, '.txt'), lambdaLog, ' ');
end

end
end
end

toc(learnTime);