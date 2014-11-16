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
%       distance
%       initT
%       fitCurve
%       optimizeParam
%       splitBezier
%       plotCubicBezier
%       detectCorners
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


    % Checks if the outline is a closed discrete curve.
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
    plot(X(1,C),X(2,C),'b*','MarkerSize',10);
    
    % Adds the first corner as the last corner with the right index.
    tempC = C;
    C = [C, C(end) + C(1) + length(X(1, C(end) : length(X(1, :)))) - 1];
    
    % Adds the points before the first corner to the end of the list.
    X = horzcat(X, X(:, 1 : tempC(1)));
    
    % Matrix to store all the cruve info for plotting.
    curves = [];
    
    % Tangent stack to maintain C1 continuity when splitting occurs.
    tangent = [];
    
    c = 1; % Counter for while loop.
    
    % Error treshold for each curve.
    errList = []; % A list with the error for each curve.
    colorFlag = 1; % For coloring adjacent curves.
    
%     xtest = linspace(0, 1, 1000);
%     ytest = sin(pi*xtest);
%     X = [];
%     X(1, :) = xtest;
%     X(2, :) = ytest;
%     C = [];
%     C(1) = 1;
%     C(2) = 1000;
%     hold off;
%     plot(xtest, ytest);
%     hold on;

    treshold = ((max(X(1, :)) + max(X(2, :))) / 2) / 10^3;
    disp(treshold);
    
    % Makes a Bezier curve between every corner point.
    while c < length(C)
        
        % Points between the current corner point and the next one.
        Xc = X(:, C(c) : C(c + 1));
        l = length(Xc(1, :));
        
        % Initial parametrization.
        ti = initT(Xc);
        
        [height, width] = size(tangent);
        
        vprev = [];
        vnext = [];
        for i = 1 : height
            if C(c) == tangent(i, 3)
                vnext = -[tangent(i, 1), tangent(i, 2)];
            end
        end
        
        for j = 1 : height
            if C(c + 1) == tangent(j, 3)
                vprev = [tangent(j, 1), tangent(j, 2)];
            end
        end
            
        
        % Initial curve.
        [P0, P1, P2, P3] = fitCurve(Xc, ti, vprev, vnext);
        
        % Optimizing parametrization.
        t = ti;
        t = optimizeParam(P0, P1, P2, P3, Xc, t);
        
        % Improved curve.
        [P0, P1, P2, P3] = fitCurve(Xc, t, vprev, vnext);
        
        % Calculating the distance between each point in the original point
        % and the Bezier curve.
        d = distance(P0, P1, P2, P3, Xc, t);
        
        % The sum of the distance over the curve.
        err = sum(d) / length(d);
        
        % Splits the curve if the error is greater than the treshold and
        % the number of points in the curve are greater than 10.
        if err > treshold && l > 8
            % Index of the split point in Xc, and v0, v3 with C1
            % continuity.
            [corner, vprev] = splitBezier(P0, P1, P2, P3, Xc, t);
            
            % Inserting the new corner point index in C, and a 1 in flag.
            temp = C;
            C = [temp(1 : c), corner + temp(c), temp(c + 1 : length(temp))];
            
            tangent = [tangent; vprev', temp(c) + corner];
            
        % Stores the curve in curves if error is below the treshold
        else
            curves = [curves; P0, P1, P2, P3, colorFlag];
            colorFlag = colorFlag + 1; % Incrementing the colorFlag.
            errList = [errList, err]; % Inserting the error.
            c = c + 1; % Jumps to next corner point.
        end
    
    end
    fprintf('Ferdig med løkke. length(C) = %i, c = %i\n', length(C), c);

    totErr = sum(errList) / length(curves(:, 1));
    fprintf('Total curve error: %f\n', totErr); % Printing total curve error.
    
    for i = 1 : length(curves(:, 1))
        plotCubicBezier([curves(i, 1), curves(i, 2)], [curves(i, 3), curves(i, 4)],...
            [curves(i, 5), curves(i, 6)], [curves(i, 7), curves(i, 8)], curves(i, 9));
    end
    
    title('Bezier curve plot');
    xlabel('x');
    ylabel('y');
    hold off;

end

function d = distance(P0, P1, P2, P3, X, ts)
%DISTANCE Calculates the distance bewteen each point of the original curve
%and the Bezier curve.
%   
%   INPUT: P0, P1, P2, P3 (Control points for the Bezier curve)
%          X (Poins describing the original curve)
%          ts (Parametrization, 0 < t < 1)
%
%   OUTPUT: d (A list with the distance between corresponding points)
%
    
    l = length(X(1, :)); % Curve length.

    XB = zeros(2, l); % Points along the Bezier curve
    sX = zeros(2,l); % Distance between the original curve and the Bezier curve in
                     % x and y direction.

    count = 1;
    for t = ts

        % Bernstein polynomials.
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;

        % Calculates each point along the Bezier curve.
        XB(:, count) = (P0*B0 + P1*B1 + P2*B2 + P3*B3)';

        count = count + 1;
    end

    for i = 2 : l - 1
        sX(:, i) = ((XB(:, i)) - X(:, i)).^2;
    end
    
    % Summing the distance in x and y.
    d = sX(1, :) + sX(2, :);

end

function t = initT(X)
%INITT Creates a initial parametrization of t based on the points
%   
%   INPUT: X (Poins describing the original curve)
%
%   OUTPUT: t (Initial parametrization of the curve, where 0 < t < 1)
%

   m = length(X(1, :)); % Curve length.
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

function [P0, P1, P2, P3] = fitCurve(X, ti, vprev, vnext)
%FITCURVE Creates the four Bezier control points from x- and y-coordinates
%by parametrization and curve fitting.
%
%   INPUT: X (Poins describing the original curve)
%          ti (Parametrization, 0 < t < 1)
%          vprev (Holds the v3 to the curve if it has to maintain C1
%                 continuity, empty otherwise)
%          vnext (Holds the v0 to the curve if it has to maintain C1
%                 continuity, empty otherwise)
%
%   OUTPUT: P0, P1, P2, P3 (Control points for the Bezier curve)
%

    % Number of points
    m = length(X(1, :));
    
    step = ceil(m/100 * 5);
    if step == 1, step = 2; end

    v0 = [X(1, step) - X(1, 1), X(2, step) - X(2, 1)];
    v0 = v0/norm(v0);
    v3 = [X(1, m - step) - X(1, m), X(2, m - step) - X(2, m)];
    v3 = v3/norm(v3);
    
    if isempty(vprev) && ~isempty(vnext)
        v0 = vnext;
    elseif isempty(vnext) && ~isempty(vprev)
        v3 = vprev;
    elseif ~isempty(vprev) && ~isempty(vnext)
        v3 = vprev;
        v0 = vnext;
    end

    
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
    
    if (A11 * A22 - A12 * A12) < 10^(-6)
        dist = X(:, 1) - X(:, m);
        k = norm(dist)/3;
        a1 = k;
        a2 = k;
    else
        a1 = (A22 * A1 - A2 * A12) / (A11 * A22 - A12 * A12);
        a2 = (A11 * A2 - A1 * A12) / (A11 * A22 - A12 * A12);
    end
    
    P1 = P0 + a1*v0;
    P2 = P3 + a2*v3;

end

function newT = optimizeParam(P0, P1, P2, P3, X, ti)
%OPTIMIZEPARAM Optimizes the parametrization of a Bezier curve by using the Newton-Raphson method.
%   
%   INPUT: P0, P1, P2, P3 (Control points for the Bezier curve)
%          X (Poins describing the original curve)
%          ti (Parametrization, 0 < t < 1)
%
%   OUTPUT: newT (A new set of t's, 0 < t < 1, obtained from Newton-Raphson
%                 method)
%
    
    l = length(X(1, :));
    oldT = zeros(1, l);
    newT = ti;
    % Points along the Bezier curve.
    XB = zeros(2, l);
    % Derivative of XB.
    XBd = zeros(2, l);
    % Second derivative of XB.
    XBdd = zeros(2, l);
    
    treshold = l/10; % Treshold 10 % of the length of the list.
    err = l; % The difference between the next parametrization and the last one.
    maxIt = 50; % Maximum number of iterations.
    it = 1; % Iteration counter.
    
    while err > treshold && it < maxIt

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
            
            % Points along the Bezier curve and it's derivatives.
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
        
        oldT = newT;
        for i = 2 : l - 1
            sX(:, i) = ((XB(:, i)) - X(:, i)).^2;

            sXd(:, i) = (XB(:, i) - X(:, i)) .* XBd(:, i);

            sXdd(:, i)= (XBd(:, i)).^2 + (XB(:, i) - X(:, i)) .* XBdd(:, i);

            if (sXdd(1, i) + sXdd(2, i)) == 0
                % Does not change the t, if the Newton-Raphson would divide
                % by zero.
            else
                newT(i)= newT(i) - ((sXd(1, i) + sXd(2, i)) / (sXdd(1, i) + sXdd(2, i)));
            end
            
        end
        
        newT = newT/newT(l); % Rescaling the new t.
        [P0, P1, P2, P3] = fitCurve(X, newT, [], []); % Calculating new Bezier control points.
        err = sum(oldT - newT)^2; % Calculating the error between the parametrizations.
        it = it + 1;
    end

end

function [split, v3] = splitBezier(P0, P1, P2, P3, X, t)
%SPLITBEZIER Splits a Bezier curve at maximum error point, and return four
%new control points for two new Bezier curves
% 
%   INPUT: P0, P1, P2, P3 (Control points for the Bezier curve)
%          X (Poins describing the original curve)
%          t (Parametrization, 0 < t < 1)
%
%   OUTPUT: split (The index of the chosen split point in X)
%           v3 (Vector for maintaining C1 continuity in the split point, where v0 = -v3)
%
    
    l = length(X(1, :));
    
    d = distance(P0, P1, P2, P3, X, t);
    
    % Finds the point where the error is greatest.
    dmax = max(d);
    imax = find(dmax == d, 1);
    
    % If split point is too close to the end points, we choose the middle
    % point.
    if imax < 5 || imax > l - 6
        imax = floor(l/2);
    end
    
    split = imax;
    
    a = X(:, split); % Split point.
    b = X(:, split + 1); % First point after split point.
    c = X(:, split - 1); % First point before split point.
    
    tempv3 = c - a;
    tempv0 = b - a;
    
    v3 = (tempv3 - tempv0) / norm(tempv3 - tempv0);

end

function plotCubicBezier(P0, P1, P2, P3, colorFlag)
%PLOTCUBICBEZIER Plots a cubic Bezier curve given by the x- and
%y-coordinated of four distinct points.
%
%   INPUT: P0, P1, P2, P3 (Control points for the Bezier curve)
%          colorFlag (A counter that decides the color of the curve)
%

    X = [P0(1), P1(1), P2(1), P3(1)];
    Y = [P0(2), P1(2), P2(2), P3(2)];
    
    % Plots the four points with filled red circles.
    plot(P0(1), P0(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P1(1), P1(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P2(1), P2(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P3(1), P3(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    
    % Plots straight, stippled, red lines between the four points.
    plot(X, Y, 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r');
    
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
    
    if mod(colorFlag, 2) == 0
        % Plotting the Bezier curve in green.
        plot(xB, yB, 'LineWidth', 1.5, 'Color', [0 0.8 0]);
    else
        % Plotting the Bezier curve in blue.
        plot(xB, yB, 'LineWidth', 1.5, 'Color', 'b');
    end
    
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
