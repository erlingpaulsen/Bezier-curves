function cubicbezier(P0x,P0y,P1x,P1y,P2x,P2y,P3x,P3y)

    X=[P0x,P1x,P2x,P3x];
    Y=[P0y,P1y,P2y,P3y];

    plot(P0x,P0y,'Marker','X','MarkerSize',20);
    hold on;
    plot(P1x,P1y,'Marker','X','MarkerSize',20);
    plot(P2x,P2y,'Marker','X','MarkerSize',20);
    plot(P3x,P3y,'Marker','X','MarkerSize',20);
    plot(X,Y);
    

    for t=0:0.001:1
        B0 = (1-t)^3;
        B1 = 3*t*(1-t)^2;
        B2 = 3*t^2*(1-t);
        B3 = t^3;

        x=P0x*B0+P1x*B1+P2x*B2+P3x*B3;
        y=P0y*B0+P1y*B1+P2y*B2+P3y*B3;


        plot(x,y);
    end
end