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

function [distance, gradient] = computeDistance( data, weights, tripletSlice, param )
    % distance: # of triplets * 2
    % gradient: total weights dimension (dimX) * # of triplets * 2

    numShapes = size(data.s, 1);
    numTriplets = size(data.ti, 1);

    dimPD = size(data.pd{1,1},2);
    dimSD = size(data.sd{1,1},2);
    dimS = size(data.s{1,1},2);
    dimX = dimSD + dimS + 3;

    Wpd = weights(1:dimPD);
    Wsd = weights(1:dimSD);
    Ws = weights(dimSD+1:dimSD+dimS);
    Wu = weights(dimSD+dimS+1);
    Wdb = weights(dimSD+dimS+2);
    Wsb = weights(dimSD+dimS+3);

    % compute saliencies & gradients on points

    pointS = cell(numShapes, 1); % # of points * 1
    gPointS = cell(numShapes, 1); % # of points * dimS+1
    shapeS = cell(numShapes, 1); % 1 * 1
    gShapeS = cell(numShapes, 1); % 1 * dimS+1
    for k = 1:numShapes
        if ~param.skipSaliencyTerm
            if ~param.useSigmoidsSaliency
                pointS{k} = data.s{k}*Ws;
                gPointS{k} = [data.s{k} zeros(size(data.s{k},1),1)];
            else
                pointS{k} = sigmoid( data.s{k}*Ws + Wsb );
                gPointS{k} = bsxfun(@times, [data.s{k} ones(size(data.s{k},1),1)], pointS{k}.*(1-pointS{k}));
            end
        else
            pointS{k} = ones(size(data.s{k},1),1);
            gPointS{k} = zeros(size(data.s{k},1), dimS+1);
        end
        shapeS{k} = sum(pointS{k});
        gShapeS{k} = sum(gPointS{k}, 1);
    end

    % compute distances & gradients on patches

    distance = zeros(numTriplets, 2);
    gradient = zeros(dimX, numTriplets, 2);
    for k = tripletSlice(:)'
        for j = 1:2
            
            %%%% distance %%%%
            
            if ~param.useSigmoidsDistance
                patchD = data.pd{k,j}*Wpd; % # of elements * 1
                gPatchD = [data.pd{k,j} zeros(size(data.pd{k,j},1))]; % # of elements * dimPD+1
                shapeD = data.sd{k,j}*Wsd; % 1 * 1
                gShapeD = [data.sd{k,j} 0]; % 1 * dimSD+1
            else
                patchD = sigmoid( (data.pd{k,j}*Wpd + Wdb) ); % # of elements * 1
                gPatchD = bsxfun(@times, [data.pd{k,j} ones(size(data.pd{k,j},1))], patchD.*(1-patchD)); % # of elements * dimPD+1
                shapeD = sigmoid( (data.sd{k,j}*Wsd + Wdb) ); % 1 * 1
                gShapeD = shapeD * (1-shapeD) * [data.sd{k,j} 1]; % 1 * dimSD+1
            end
            
            %%%% saliency %%%%
            
            % slice data
            
            ss = pointS{data.ti(k,1)};   % point saliencies of first shape (A)
            st = pointS{data.ti(k,1+j)}; % point saliencies of second shape (B or C)
            zs = shapeS{data.ti(k,1)};   % shape saliency
            zt = shapeS{data.ti(k,1+j)};

            gss = gPointS{data.ti(k,1)}; % gradients of terms above
            gst = gPointS{data.ti(k,1+j)};
            gzs = gShapeS{data.ti(k,1)};
            gzt = gShapeS{data.ti(k,1+j)};
            
            ms = max(1, data.ms{k,j}); % 1 * # of points
            mt = max(1, data.mt{k,j});
            
            % compute saliency
            
            ss = ss ./ ms'; % normalize by # of elements (unchanged for unmatched points)
            st = st ./ mt';

            nss = ss / zs; % normalized point saliency: # of points * 1
            nst = st / zt;
            
            patchS = ( data.as{k,j}*nss + data.at{k,j}*nst ) / 2; % # of elements * 1
            unmatchS = ( data.us{k,j}*nss + data.ut{k,j}*nst ) / 2; % 1 * 1
            
            % compute gradient of saliency
            
            pds = patchD'*data.as{k,j}; % pre-compute this term first: 1 * # of points
            pdt = patchD'*data.at{k,j};
            
            gms = (pds./ms)*gss / zs - pds*ss*gzs / (zs*zs); % 1 * dimS+1
            gmt = (pdt./mt)*gst / zt - pdt*st*gzt / (zt*zt);
            gMatchS = ( gms + gmt ) / 2; % 1 * dimS(+1)
            
            gus = (data.us{k,j}./ms)*gss / zs - data.us{k,j}*ss*gzs / (zs*zs); % 1 * dimS+1
            gut = (data.ut{k,j}./mt)*gst / zt - data.ut{k,j}*st*gzt / (zt*zt);
            gUnmatchS = ( gus + gut ) / 2; % 1 * dimS+1
            
            %%%% put everything together %%%%
            
            dMatch = 0;
            if ~param.skipGlobalDistance
                dMatch = dMatch + shapeD;
            end
            if ~param.skipElementDistance
                dMatch = dMatch + patchS'*patchD;
            end
            dUnmatch = Wu * unmatchS;
            dAll = 0;
            if ~param.skipMatchedTerm
                dAll = dAll + dMatch;
            end
            if ~param.skipUnmatchedTerm
                dAll = dAll + dUnmatch;
            end
            distance(k, j) = dAll;

            if ~param.skipMatchedTerm
                gWpd = patchS'*gPatchD; % 1 * dimPD+1
                gWsd = gShapeD; % 1 * dimSD+1
                gWd = zeros(1, dimSD+1);
                if ~param.skipGlobalDistance
                    gWd = gWd + gWsd;
                end
                if ~param.skipElementDistance
                    gWd = gWd + [gWpd(1:dimPD) zeros(1, dimSD-dimPD) gWpd(end)];
                end
            else
                gWd = zeros(1, dimSD+1);
                gMatchS = zeros(1, dimS+1);
            end
            if ~param.skipUnmatchedTerm
                gWu = unmatchS; % 1 * 1
            else
                gUnmatchS = zeros(1, dimS+1);
                gWu = 0;
            end
            gWs = gMatchS + Wu*gUnmatchS; % 1 * dimS+1
            
            gAll = zeros(dimX, 1);
            gAll(1:dimSD) = gWd(1:dimSD)';
            gAll(dimSD+1:dimSD+dimS) = gWs(1:dimS)';
            gAll(dimSD+dimS+1) = gWu;
            gAll(dimSD+dimS+2) = gWd(end);
            gAll(dimSD+dimS+3) = gWs(end);
            
            gradient(:,k,j) = gAll;
        end
    end
end

