function [P0, P1, P2, P3] = fitCurve(X, ti)
%FITCURVE Creates the four Bezier control points from x- and y-coordinates
%by parametrization and curve fitting.
%
%   INPUT: X (Matrix with x-coordinates in first column and y-coordinated
%   in second column)
%
%   OUTPUT: Four Bezier control points.

    % Number of points
    m = length(X(1, :));
    
    v0 = [X(1, 2) - X(1, 1), X(2, 2) - X(2, 1)];
    v0 = v0/norm(v0);
    v3 = [X(1, m) - X(1, m - 1), X(2, m) - X(2, m - 1)];
    v3 = v3/norm(v3);
    
    P0 = [X(1, 1), X(2, 1)];
    P3 = [X(1, m), X(2, m)]; 
    
    A1 = 0;
    A12 = 0;
    A2 = 0;
    A11 = 0;
    A22 = 0;
    
    % Counter
    i = 1;
    
    for t = ti
        
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;
        
        Z = X(:, i)' - P0*B0 - P0*B1 - P3*B2 - P3*B3;
        
        A1 = A1 + dot(Z, v0)*B1;
        A12 = A12 + dot(v0, v3)*B1*B2;
        A2 = A2 + dot(Z, v3)*B2;
        A11 = A11 + dot(v0, v0) * B1^2;
        A22 = A22 + dot(v3, v3) * B2^2;
        
        i = i + 1;
        
    end
    
    a1 = (A22 * A1 - A2 * A12) / (A11 * A22 - A12 * A12);
    a2 = (A11 * A2 - A1 * A12) / (A11 * A22 - A12 * A12);
    
    P1 = P0 + a1*v0;
    P2 = P3 + a2*v3;


end