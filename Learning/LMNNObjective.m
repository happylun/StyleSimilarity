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

function [E, G] = LMNNObjective( X, data, trainSet, lambda, param )
    % X: weights to be learned (dimX * 1)
    % data: learning data (triplets, distances, saliencies, ...)
    % trainSet: triplets slice for training set
    % lambda: regularization weight
    % param: global parameters
    
    % params
    mu = 1.0;
    
    [distance, gradient] = computeDistance(data, X, trainSet, param);
    
    % swap B and C if answer is C
    swapIndex = data.ta == 2;
    distance(swapIndex, :) = distance(swapIndex, [2 1]);
    gradient(:, swapIndex, :) = gradient(:, swapIndex, [2 1]);
    
    % L1 regularization
    R = abs(X)'*lambda;
    dRdX = sign(X) .* lambda;
    
    costAB = distance(trainSet,1); % # of triplets * 1
    costAC = distance(trainSet,2); % # of triplets * 1
    gradientAB = gradient(:, trainSet, 1); % dimX * # of triplets
    gradientAC = gradient(:, trainSet, 2); % dimX * # of triplets
    confidence = data.tw(trainSet);
    N = ( 1+costAB-costAC >= 0 ); % triplets which trigger the hinge lost
    E = (1-mu) * confidence' * costAB + mu * confidence' * max(0, 1+costAB-costAC) + R;
    G = (1-mu) * gradientAB * confidence + mu * (gradientAB(:,N)-gradientAC(:,N)) * confidence(N) + dRdX;
    
    %{
    pX = costAB - costAC;
    pY = max(0, 1+pX);
    figure(1);
    scatter(pX, pY, '.');
    line([0, 0], [0, 4], 'LineStyle', '--', 'Color', 'red');
    axis([-4, 4, 0, 4]);
    fprintf('%d\t', nnz(N));
    %}
    
    %{
    figure(1);
    clf;
    plot(costAB, 'g');
    hold on;
    plot(costAC, 'r');
    axis([0, 140, 0, 6]);
    fprintf('%d\t', nnz(N));
    %}
end

