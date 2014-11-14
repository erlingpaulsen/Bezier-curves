% The script genOutline generates a traced outline of a bitmap image.
% The image is assumed to consist of a single object whose outline is
% desired.

I = imread('elephant.jpg'); % Read the image file. image(I) displays the image.

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
C = CornerDet(X);


% Plots the discrete curve.
plot(X(1,[1:end,1]),X(2,[1:end,1]),'k-','LineWidth',1);
axis equal;

% Overlays the found corner points.
hold on;
plot(X(1,C),X(2,C),'r*','MarkerSize',8);
hold off;