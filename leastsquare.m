function leastsquare()
    x=[-2.0000000000000000 -1.9900095166399880 -1.9604352202265733 -1.9124473886531921 -1.8479270408896213 -1.7693657878779678 -1.6797334297990107 -1.5823205857872562 -1.4805650134826276 -1.3778711689435199 -1.2774329231045609 -1.1820691689841958 -1.0940813343339346 -1.0151405971764766 -0.9462109540099681 -0.8875123034485414 -0.8385254915624213 -0.7980389416901280 -0.7642341894453175 -0.7348054902675931 -0.7071067811865477 -0.6783177647010343 -0.6456198244848970 -0.6063719385004512 -0.5582767551049996 -0.4995275418736065 -0.4289277750600726 -0.3459766513932762 -0.2509156895939807 -0.1447337423783132 -0.0291300417718178  0.0935627759435417  0.2204915028125257  0.3484248994685338  0.4739181455454290  0.5934901854586080  0.7038042352542520  0.8018415342404277  0.8850587907655991  0.9515206646401961  0.9999999999999998];
    y=[-0.0000000000000002  0.0785196081826872  0.1569174458721380  0.2350691476799097  0.3128452182202336  0.3901086156222856  0.4667125105489297  0.5424982747558325  0.6172937494799466  0.6909118393701043  0.7631494723325429  0.8337869596396903  0.9025877840308228  0.9692988364166278  1.0336511142920473  1.0953608871751803  1.1541313264407198  1.2096545889237820  1.2616143357542522  1.3096886601626476  1.3535533905932737  1.3928857284824336  1.4273681736161741  1.4566926841738781  1.4805650134826271  1.4987091612341636  1.5108718735219759  1.5168271235984572  1.5163805037762810  1.5093734584351104  1.4956872886612853  1.4752468606462859  1.4480239525869567  1.4140401784394196  1.3733694314378115  1.3261397957431493  1.2725348808683927  1.2127945405527212  1.1472149454394973  1.0761479871471193  1.0000000000000004];
    ti=[0.0000000000000000  0.0250000000000000  0.0500000000000000  0.0750000000000000  0.1000000000000000  0.1250000000000000  0.1500000000000000  0.1750000000000000  0.2000000000000000  0.2250000000000000  0.2500000000000000  0.2750000000000000  0.3000000000000000  0.3250000000000000  0.3500000000000000  0.3750000000000000  0.4000000000000000  0.4250000000000000  0.4500000000000000  0.4750000000000000  0.5000000000000000  0.5250000000000000  0.5500000000000000  0.5750000000000000  0.6000000000000000  0.6250000000000000  0.6500000000000000  0.6750000000000000  0.7000000000000000  0.7250000000000000  0.7500000000000000  0.7750000000000000  0.8000000000000000  0.8250000000000000  0.8500000000000000  0.8750000000000000  0.9000000000000000  0.9250000000000000  0.9500000000000000  0.9750000000000000  1.0000000000000000];

    m = length(x);
    %Ha med dette dersom man vil diskretisere tiden vektet p� avstand
    %mellom punkter.
    %d = zeros(1, m);
    %for i=1:m-1
    %   d(i+1) = d(i)+(sqrt((x(i+1)-x(i))^2+(y(i+1)-y(i))^2));
    %end
    %ti = d/d(m);
    teller = 1;
    %Endre p� inkrementet i forl�kken for � bruke f�rre punkter i utregning
    %av Bezier.
    for i=1:1:m
        xs(teller) = x(i);
        ys(teller) = y(i);
        ts(teller) = ti(i);
        teller = teller + 1;
    end
    ms = length(xs);
    disp(ms);
    if xs(ms) ~= x(m);
        xs(ms+1) = x(m);
        ys(ms+1) = y(m);
        ts(ms+1) = ti(m);
        ms = ms+1;
    end
    %ts = linspace(0, 1, ms);
    P0 = [xs(1), ys(1)];
    P3 = [xs(ms), ys(ms)];
    v0 = [xs(2)-xs(1), ys(2)-ys(1)];
    v0 = v0/norm(v0);
    v3 = [xs(ms-1)-xs(ms), ys(ms-1)-ys(ms)];
    v3 = v3/norm(v3);
    teller = 1;
    A1 = 0;
    A11 = 0;
    A12 = 0;
    A22 = 0;
    A2 = 0;
    for t = ts;
        b0 = (1-t)^3;
        b1 = 3*t*((1-t)^2);
        b2 = 3*(t^2)*(1-t);
        b3 = t^3;
    
        A = [xs(teller), ys(teller)] - P0*b0 - P0*b1 - P3*b2 - P3*b3;
        A1 = A1 + dot(A, v0)*b1;
        A11 = A11 + dot(v0, v0)*(b1^2);
        A12 = A12 + dot(v0,v3)*b1*b2;
        A22 = A22 + dot(v3, v3)*(b2^2);
        A2 = A2 + dot(A, v3)*b2;
        teller = teller + 1;
    end
        
    a1 = (A22*A1 - A2*A12)/(A11*A22 - A12*A12);
    a2 = (A11*A2 - A1*A12)/(A11*A22 - A12*A12);
    P1 = P0 + a1*v0;
    P2 = P3 + a2*v3;
    E = 0;
    for i = 1:m
        b0 = (1-ti(i))^3;
        b1 = 3*ti(i)*(1-ti(i))^2;
        b2 = 3*(ti(i)^2)*(1-ti(i));
        b3 = ti(i)^3;
        E = E + (norm([x(i), y(i)]-P0*b0-P1*b1-P2*b2-P3*b3))^2;
    end
    E = E/m;
    disp(E);
    
    plot(x,y);
    hold on;
    cubicbezier(P0(1), P0(2), P1(1), P1(2), P2(1), P2(2), P3(1), P3(2));
end