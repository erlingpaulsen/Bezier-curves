function convertToBezier(filename)
%CONVERTTOBEZIER Converts a shape from an image to a set of Bezier
%curves, and plots both the original curve and the Bezier curves. 
%The image is assumed to consist of a single object whose outline is desired.
%
%   INPUT: filename (A string containing the filename of a picture, e.g 'batman.jpg')
%
%   CONVERTTOBEZIER(filename) will convert the shape found in
%   a picture with the specified filename to Bezier curves if the
%   picture is located in the root.
%   CONVERTTOBEZIER uses several local functions:
%       initT.
%       fitCurve.
%       splitBezier.
%       plotCubicBezier.
%       detectCorners.
%       
%   Write 'help convertToBezier>[local function name]' to see their documentation.
%

    I = imread(filename); % Read the image file. image(I) displays the image.

    BW = im2bw(I); % Converts the image to a black and white image. Can display
                   % the image using imshow(BW);

    BW = ~BW; % Switches black and white. Skip if your image has a black background.

    BW = imfill(BW,'holes'); % Fills any holes in the image.

    BW1 = bwperim(BW); % Detects the perimeter/outline of the image.

    [I,J] = find(BW1,1,'first'); % Finds the starting point for the tracing algorithm.

    X = bwtraceboundary(BW1,[I,J],'W')'; % Traces the image clockwise.

    % Transform from row and column coordinates to traditional x and y;
    X = [X(2,:);-X(1,:)]; 
    m = min(X(2,:));
    X(2,:) = X(2,:) - m; 


    % Check if the outline is a closed discrete curve.
    if norm(X(:,1)-X(:,end),inf) < eps
        X = X(:,1:end-1); % Remove the last point so every point is included 
                          % just once.
    else
        error('The outline is not a closed discrete curve');
    end


    s = 2; % Smoothing parameter. Higher value results in more smoothing.

    % Smooths the outline.
    Xc = X;
    for i = 1:s
    X = X + Xc(:,[1+i:end,1:i]);
    X = X + Xc(:,[end-(i-1):end,1:end-i]);
    end
    X = X/(2*s+1);

    % Finds the corner indices of the image. It may be necessary to adjust the
    % parameters of the algorithm to get a satisfactory result.
    C = detectCorners(X);


    % Plots the discrete curve.
    plot(X(1,[1:end,1]),X(2,[1:end,1]),'k-','LineWidth',1);
    axis equal;

    % Overlays the found corner points.
    hold on;
    plot(X(1,C),X(2,C),'r*','MarkerSize',8);
    
    %fprintf('X0 = %.1f, Y0 = %.1f, Xn = %.1f, Yn = %.1f\n', X(1, 1), X(2, 1), X(1, length(X(1, :))), X(2, length(X(1, :))));
    X = horzcat(X, X(:, 1 : C(1)));
    C = [C, C(length(C)) + C(1)];
    %fprintf('X0 = %.1f, Y0 = %.1f, Xn = %.1f, Yn = %.1f\n', X(1, 1), X(2, 1), X(1, length(X(1, :))), X(2, length(X(1, :))));
    
    c = 1;
    treshold = 0.1;
    
    % Makes a Bezier curve between every corner point.
    while c < length(C)
        
        Xc = X(:, C(c) : C(c + 1));
        ti = initT(Xc);

        [P0, P1, P2, P3] = fitCurve(Xc, ti);
        
        % Optimizing parametrization.
        t = ti;
        for p = 1 : 5
            t = optimizeParam(P0, P1, P2, P3, Xc, t);
        end
        
        % Improved curve.
        [P0, P1, P2, P3] = fitCurve(Xc, t);
        
        % Calculating the distance between each point in the original point
        % and the Bezier curve.
        d = distance(P0, P1, P2, P3, Xc, t);
        % The sum of the distance over the curve.
        err = sqrt(sum(d(1, :) + d(2, :))) / length(d(1, :));
        
        % Splits the curve if the error is greater than the treshold
        if err > treshold
            % Index of the split point in Xc
            corner = splitBezier(P0, P1, P2, P3, Xc, t);
            
            % Inserting the new corner point in C.
            temp = C;
            C = [temp(1 : c), corner + temp(c), temp(c + 1 : length(temp))];
            
        % Plots the curve if error is below the treshold
        else
            plotCubicBezier(P0, P1, P2, P3);
            c = c + 1;
        end
    
    end
    fprintf('Ferdig med l�kke. length(C) = %i, c = %i\n', length(C), c);
    
    hold off;

end

function sX = distance(P0, P1, P2, P3, X, ts)
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

function t = initT(X)
%INITT Creates a initial parametrization of t based on the points
%   
%   INPUT: X (Poins describing the curve)

   m = length(X(1, :));
   d = zeros(1, m);

   %initial t
   d(1) = 0;
   for i = 1 : m - 1
       % List of cummulative length between points.
       d(i + 1) = d(i) + (sqrt((X(1, i + 1) - X(1, i))^2 + (X(2, i + 1) - X(2, i))^2));
   end
   
   % Initial t, 0 < t < 1
   t = d/d(m);
    
end

function [P0, P1, P2, P3] = fitCurve(X, ti)
%FITCURVE Creates the four Bezier control points from x- and y-coordinates
%by parametrization and curve fitting.
%
%   INPUT: X (Matrix with x-coordinates in first column and y-coordinated
%   in second column)
%
%   OUTPUT: Four Bezier control points.

    % Number of points
    %fprintf('(X0, Y0) = (%.1f, %.1f), (Xn, Yn) = (%.1f, %.1f), length(X) = %i\n', X(1, 1), X(2, 1), X(1, length(X(1, :))), X(2, length(X(1, :))), length(X(1, :)));
    m = length(X(1, :));
    
    v0 = [X(1, 2) - X(1, 1), X(2, 2) - X(2, 1)];
    v0 = v0/norm(v0);
    v3 = [X(1, m - 1) - X(1, m), X(2, m - 1) - X(2, m)];
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
        
        A1 = A1 + dot(Z, v0) * B1;
        A12 = A12 + dot(v0, v3) * B1 * B2;
        A2 = A2 + dot(Z, v3) * B2;
        A11 = A11 + dot(v0, v0) * B1^2;
        A22 = A22 + dot(v3, v3) * B2^2;
        
        i = i + 1;
        
    end
    
    a1 = (A22 * A1 - A2 * A12) / (A11 * A22 - A12 * A12);
    a2 = (A11 * A2 - A1 * A12) / (A11 * A22 - A12 * A12);
    
    P1 = P0 + a1*v0;
    P2 = P3 + a2*v3;

end

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

        sXd(:, i) = 2*(XB(:, i) - X(:, i)) .* XBd(:, i);

        sXdd(:, i)= (XBd(:, i)).^2 + (XB(:, i) - X(:, i)) .* XBdd(:, i);

        newT(i)= newT(i) - ((sXd(1, i) + sXd(2, i)) / (sXdd(1, i) + sXdd(2, i)));   
    end

end

function split = splitBezier(P0, P1, P2, P3, X, t)
%SPLITBEZIER Splits a Bezier curve at maximum error point, and return four
%new control points for two new Bezier curves
%   Detailed explanation goes here
    
    l = length(X(1, :));
    
    d = distance(P0, P1, P2, P3, X, t);
    
    dmax = max(d(1, :) + d(2, :));
    imax = find(dmax == (d(1, :) + d(2, :)));
    
    split = imax;
%     
%     XR = X(:, 1 : imax);
%     XS = X(:, imax : l);
% 
%     tR = initT(XR);
%     tS = initT(XS);
%     
%     [R0, R1, R2, R3] = fitCurve(XR, tR);
%     [S0, S1, S2, S3] = fitCurve(XS, tS);

end

function plotCubicBezier(P0, P1, P2, P3)
%PLOTCUBICBEZIER Plots a cubic Bezier curve given by the x- and
%y-coordinated of four distinct points.
%
%   INPUT: Pix (x-coordinate to point number i)
%          Piy (y-coordinate to point number i)
%          Where i = 0, 1, 2, 3
%

    X = [P0(1), P1(1), P2(1), P3(1)];
    Y = [P0(2), P1(2), P2(2), P3(2)];
    
    % Plots the four points with filled red circles.
    plot(P0(1), P0(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P1(1), P1(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P2(1), P2(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P3(1), P3(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    
    % Plots straight, stippled, red lines between the four points.
    plot(X, Y, 'LineStyle', '--', 'LineWidth', 1, 'Color', 'r');
    
    % The step size for 0 < t < 1.
    step = 0.0001;
    
    % Vectors to store the points along the Bezier curve.
    xB = zeros(1, 1/step + 1);
    yB = zeros(1, 1/step + 1);
    
    % Using Bernstein polynomials to generate points along the Bezier
    % curve.
    for t = 0 : step : 1
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;
        
        xB(int32(t * (1/step) + 1)) = P0(1)*B0 + P1(1)*B1 + P2(1)*B2 + P3(1)*B3;
        yB(int32(t * (1/step) + 1)) = P0(2)*B0 + P1(2)*B1 + P2(2)*B2 + P3(2)*B3;
        
    end
    
    % Plotting the Bezier curve in black.
    plot(xB, yB, 'LineWidth', 1.5, 'Color', 'k');
    axis equal;

end

function C = detectCorners(X)
% DETECTCORNERS detects corners of traced outlines using the SAM04 algorithm. 
% The outline is assumed to constitute a discrete closed curve where each
% point is included just once.
%
% INPUT: X (A traced outline forming a discrete closed curve)
%
% OUTPUT: C (The indices of X that consitute corner points)
%

    n = length(X);

    % Sets the parameters of the algorithm. Increasing D and R will give the
    % same number of corners or fewer corners.
    L = 80; % Controls the scale at which corners are measured.
    R = 30; % Controls how close corners can appear.
    D = 14; % The lower bound for the corner metric. Corner candidates with 
            % lower metric than this are rejected.


    % Finds corner candidates        
    d = zeros(1,n);
    for i = 1:n
        if i+L <= n
            k = i+L;
            index = i+1:k-1;
        else
            k = i+L-n;
            index = [i+1:n,1:k-1];
        end

        M = X(:,k)-X(:,i);

        if M(1) == 0
            dCand = abs(X(1,index)-X(1,i));
        else
            m = M(2)/M(1);
            dCand = abs(X(2,index)-m*X(1,index)+m*X(1,i)-X(2,i))/sqrt(m^2+1);
        end

        [Y,I] = max(dCand);
        if Y > d(index(I))
            d(index(I)) = Y;
        end
    end

    % Rejects candidates which do not meet the lower metric bound D.
    index = d < D;
    d(index) = 0;
    C = 1:n;
    index = ~logical(index);
    C = C(index);


    % Rejects corners that are too close to a corner with larger metric.
    l = length(C);
    j = 1;
    while j < l
        if abs(C(j)-C(j+1)) <= R
            if d(C(j)) > d(C(j+1))
                C = C([1:j,j+2:l]);
            else
                C = C([1:j-1,j+1:l]);
            end
            l = l-1;
        else
            j = j+1;
        end
    end

    if l > 1 && abs(C(1)+n-C(end)) <=R
        if d(C(end)) > d(C(1))
            C = C(2:end);
        else
            C = C(1:end-1);
        end
    end
end
