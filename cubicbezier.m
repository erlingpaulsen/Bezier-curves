function cubicbezier(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y)

    X = [P0x, P1x, P2x, P3x];
    Y = [P0y, P1y, P2y, P3y];
    
    plot(P0x, P0y, 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    hold on;
    plot(P1x, P1y, 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P2x, P2y, 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    plot(P3x, P3y, 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    
    plot(X, Y, 'LineStyle', '--', 'LineWidth', 1, 'Color', 'r');
    
    step = 0.0001;
    
    xB = zeros(1, 1/step + 1);
    yB = zeros(1, 1/step + 1);
    for t = 0 : step : 1
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;
        
        xB(int32(t * (1/step) + 1)) = P0x*B0 + P1x*B1 + P2x*B2 + P3x*B3;
        yB(int32(t * (1/step) + 1)) = P0y*B0 + P1y*B1 + P2y*B2 + P3y*B3;
        
    end
    plot(xB, yB, 'LineWidth', 1.5, 'Color', 'k');
    
end