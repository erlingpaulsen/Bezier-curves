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
    splitCount = 0; % Counts number of splits.
    
    %xtest = linspace(0, 1, 1000);
    %ytest = sin(pi*xtest);
%     xtest = [-2.0000000000000000 -1.9900095166399880 -1.9604352202265733 -1.9124473886531921 -1.8479270408896213 -1.7693657878779678 -1.6797334297990107 -1.5823205857872562 -1.4805650134826276 -1.3778711689435199 -1.2774329231045609 -1.1820691689841958 -1.0940813343339346 -1.0151405971764766 -0.9462109540099681 -0.8875123034485414 -0.8385254915624213 -0.7980389416901280 -0.7642341894453175 -0.7348054902675931 -0.7071067811865477 -0.6783177647010343 -0.6456198244848970 -0.6063719385004512 -0.5582767551049996 -0.4995275418736065 -0.4289277750600726 -0.3459766513932762 -0.2509156895939807 -0.1447337423783132 -0.0291300417718178  0.0935627759435417  0.2204915028125257  0.3484248994685338  0.4739181455454290  0.5934901854586080  0.7038042352542520  0.8018415342404277  0.8850587907655991  0.9515206646401961  0.9999999999999998];
%     ytest = [-0.0000000000000002  0.0785196081826872  0.1569174458721380  0.2350691476799097  0.3128452182202336  0.3901086156222856  0.4667125105489297  0.5424982747558325  0.6172937494799466  0.6909118393701043  0.7631494723325429  0.8337869596396903  0.9025877840308228  0.9692988364166278  1.0336511142920473  1.0953608871751803  1.1541313264407198  1.2096545889237820  1.2616143357542522  1.3096886601626476  1.3535533905932737  1.3928857284824336  1.4273681736161741  1.4566926841738781  1.4805650134826271  1.4987091612341636  1.5108718735219759  1.5168271235984572  1.5163805037762810  1.5093734584351104  1.4956872886612853  1.4752468606462859  1.4480239525869567  1.4140401784394196  1.3733694314378115  1.3261397957431493  1.2725348808683927  1.2127945405527212  1.1472149454394973  1.0761479871471193  1.0000000000000004];
%     X = [];
%     X(1, :) = xtest;
%     X(2, :) = ytest;
%     C = [];
%     C(1) = 1;
%     %C(2) = 1000;
%     C(2) = 41;
%     hold off;
%     plot(xtest, ytest);
%     hold on;

    treshold = ((max(X(1, :)) + max(X(2, :))) / 2) / 10^3;
    
    % Makes a Bezier curve between every corner point.
    while c < length(C)
        
        % Points between the current corner point and the next one.
        Xc = X(:, C(c) : C(c + 1));
        l = length(Xc(1, :)); % Number of points.
        
        % Initial parametrization.
        ti = initT(Xc);
        
        [height, width] = size(tangent);
        
        vprev = [];
        vnext = [];
        
        % Checks if a corner point is a split point, if so v0 and v3 are
        % given, to maintain C1 continuity.
        for i = 1 : height
            if C(c) == tangent(i, 3)
                vnext = -[tangent(i, 1), tangent(i, 2)];
            end
            
            if C(c + 1) == tangent(i, 3)
                vprev = [tangent(i, 1), tangent(i, 2)];
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
            splitCount = splitCount + 1;
            
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
    
    % Sum of error over all curves divided by number of curves.
    totErr = sum(errList) / length(curves(:, 1)); 
    
    % Plotting all the Bezier curves.
    for i = 1 : length(curves(:, 1))
        plotCubicBezier([curves(i, 1), curves(i, 2)], [curves(i, 3), curves(i, 4)],...
            [curves(i, 5), curves(i, 6)], [curves(i, 7), curves(i, 8)], curves(i, 9));
    end
    
    % Printing a table of results.
    fprintf('----------------------------- RESULTS -------------------------------\n');
    fprintf('Number of splits: %i\n', splitCount);
    fprintf('Number of curves: %i\n', length(curves(:, 1)));
    fprintf('Total curve error: %f\n\n', totErr);
    fprintf('Discription of plot:\n');
    fprintf('    - Blue star: Original corner point\n');
    fprintf('    - Red dots: Bezier curve control point\n');
    fprintf('    - Red stippled line: Straight line between control points\n');
    fprintf('    - Thin black line: Original curve\n');
    fprintf('    - Blue or green line: Bezier curve\n');
    fprintf('----------------------------------------------------------------------\n');
    
    title('Bezier curve plot');
    xlabel('x');
    ylabel('y');
    fig = figure(1);
    set(fig, 'Position', [0, 100, 1200, 800]) % Plot size and position.
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
    
    % If the split point is too close to the end points, we choose the middle
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
    
    % Creates start and end vectors for the split point, where v0 = -v3.
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
    
    % Plots the four control points with filled red circles.
    plot(P0(1), P0(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
    plot(P1(1), P1(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
    plot(P2(1), P2(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
    plot(P3(1), P3(2), 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
    
    % Plots straight, stippled, red lines between the control points.
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
