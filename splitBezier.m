function [R0, R1, R2, R3, S0, S1, S2, S3] = splitBezier(P0, P1, P2, P3, X)
%SPLITBEZIER Splits a Bezier curve at maximum error point, and return four
%new control points for two new Bezier curves
%   Detailed explanation goes here
    
    d = distance(P0, P1, P2, P3, X);
    
    dmax = max(d(1, :) + d(2, :));
    imax = find(dmax == (d(1, :) + d(2, :)));
    
    XR = X(:, 1 : imax);
    XS = X(:, imax : l);

    tR = initT(XR);
    tS = initT(XS);
    
    [R0, R1, R2, R3] = fitCurve(XR, tR);
    [S0, S1, S2, S3] = fitCurve(XS, tS);

end