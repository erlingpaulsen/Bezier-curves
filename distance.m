function sX = distance(P0, P1, P2, P3, X)
%DISTANCE Summary of this function goes here
%   Detailed explanation goes here
    
    l = length(X(1, :));

    XB = zeros(2, l);
    sX = zeros(2,l);

    count = 1;
    for t = ts

        % Bernstein polynomials.
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;

        XB(:, count) = (P0*B0 + P1*B1 + P2*B2 + P3*B3)';

        count = count + 1;
    end

    for i = 2 : l - 1
        sX(:, i) = ((XB(:, i)) - X(:, i)).^2;
    end

end