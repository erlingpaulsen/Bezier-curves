function newT = optimizeParam(P0, P1, P2, P3, X, ti)
%OPTIMIZEPARAM Optimizes the parametrization of a Bezier curve.
%   Detailed explanation goes here
    
    l = length(X(1, :));
    newT = ti;
    % Points along the Bezier curve.
    XB = zeros(2, l);
    % Derivative of XB.
    XBd = zeros(2, l);
    % Second derivative of XB.
    XBdd = zeros(2, l);
    
    treshold = 0.1;
    maxIt = 50;

    count = 1;
    for t = ti

        % Bernstein polynomials.
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;

        % First derivative of Bernstein polynomials.
        B0d = 3*(1 - t)^2;
        B1d = 6 * t * (1 - t);
        B2d = 3 * t^2;

        % Second derivative of Bernstein polynomials.
        B0dd = 6*(1 - t);
        B1dd = 6 * t ;

        XB(:, count) = (P0*B0 + P1*B1 + P2*B2 + P3*B3)';
        XBd(:, count) = ((P1 - P0)*B0d + (P2 - P1)*B1d + (P3 - P2)*B2d)';
        XBdd(:, count) = ((P2 - 2*P1 + P0)*B0dd + (P3 - 2*P2 + P1)*B1dd)';

        count = count + 1;

    end

    % Distance between original curve and our parametrization.
    sX=zeros(2,l);
    % First derivative of this distance.
    sXd=zeros(2,l);
    % Second derivative of this distance.
    sXdd=zeros(2,l);

    for i = 2 : l - 1
        sX(:, i) = ((XB(:, i)) - X(:, i)).^2;

        sXd(:, i) = (XB(:, i) - X(:, i)) .* XBd(:, i);

        sXdd(:, i)= (XBd(:, i)).^2 + (XB(:, i) - X(:, i)) .* XBdd(:, i);
        
        if (sXdd(1, i) + sXdd(2, i)) == 0
            disp('feil');
           return;
        end

        newT(i)= newT(i) - ((sXd(1, i) + sXd(2, i)) / (sXdd(1, i) + sXdd(2, i)));   
    end

end