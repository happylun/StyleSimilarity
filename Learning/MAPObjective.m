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

function [E, G] = MAPObjective( X, data, trainSet, lambda, param )
    % X: weights to be learned (dimX * 1)
    % data: learning data (triplets, distances, saliencies, ...)
    % trainSet: triplets slice for training set
    % lambda: regularization weight
    % param: global parameters
    
    [distance, gradient] = computeDistance(data, X, trainSet, param);
    confidence = data.tw(:, 1:2); % # of triplets * 2
    
    % swap B and C if answer is C
    swapIndex = data.ta == 2;
    distance(swapIndex, :) = distance(swapIndex, [2 1]);
    gradient(:, swapIndex, :) = gradient(:, swapIndex, [2 1]);
    confidence(swapIndex, :) = confidence(swapIndex, [2 1]);
    
    % L1 regularization
    R = abs(X)'*lambda;
    dRdX = sign(X) .* lambda;
    
    % L2 regularization
    %R = (X.^2)'*lambda;
    %dRdX = X .* lambda * 2;
    
    deltaCost = distance(trainSet, 2) - distance(trainSet, 1); % # of triplets * 1
    deltaGradient = gradient(:, trainSet, 2) - gradient(:, trainSet, 1); % dimX * # of triplets
    confidence = confidence(trainSet, :);
    
    pBC = sigmoid(deltaCost);
    pCB = 1 - pBC;
    E = confidence(:,1)' * ( -log(pBC) ) + confidence(:,2)' * ( -log(pCB) ) + R;
    G = deltaGradient * (confidence(:,1).*(pBC-1) + confidence(:,2).*pBC) + dRdX;
    
    %{
    pX = -deltaCost;
    pY = -log(1./(1+exp(pX)));
    figure(1);
    scatter(pX, pY, '.');
    line([0, 0], [0, 1.5], 'LineStyle', '--', 'Color', 'red');
    axis([-1.5, 1.5, 0, 1.5]);
    fprintf('%d\t', nnz(pX<0));
    %}
    
end

