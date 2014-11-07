function initparametrization()
    
    x=[-2.0000000000000000 -1.9900095166399880 -1.9604352202265733 -1.9124473886531921 -1.8479270408896213 -1.7693657878779678 -1.6797334297990107 -1.5823205857872562 -1.4805650134826276 -1.3778711689435199 -1.2774329231045609 -1.1820691689841958 -1.0940813343339346 -1.0151405971764766 -0.9462109540099681 -0.8875123034485414 -0.8385254915624213 -0.7980389416901280 -0.7642341894453175 -0.7348054902675931 -0.7071067811865477 -0.6783177647010343 -0.6456198244848970 -0.6063719385004512 -0.5582767551049996 -0.4995275418736065 -0.4289277750600726 -0.3459766513932762 -0.2509156895939807 -0.1447337423783132 -0.0291300417718178  0.0935627759435417  0.2204915028125257  0.3484248994685338  0.4739181455454290  0.5934901854586080  0.7038042352542520  0.8018415342404277  0.8850587907655991  0.9515206646401961  0.9999999999999998];
    y=[-0.0000000000000002  0.0785196081826872  0.1569174458721380  0.2350691476799097  0.3128452182202336  0.3901086156222856  0.4667125105489297  0.5424982747558325  0.6172937494799466  0.6909118393701043  0.7631494723325429  0.8337869596396903  0.9025877840308228  0.9692988364166278  1.0336511142920473  1.0953608871751803  1.1541313264407198  1.2096545889237820  1.2616143357542522  1.3096886601626476  1.3535533905932737  1.3928857284824336  1.4273681736161741  1.4566926841738781  1.4805650134826271  1.4987091612341636  1.5108718735219759  1.5168271235984572  1.5163805037762810  1.5093734584351104  1.4956872886612853  1.4752468606462859  1.4480239525869567  1.4140401784394196  1.3733694314378115  1.3261397957431493  1.2725348808683927  1.2127945405527212  1.1472149454394973  1.0761479871471193  1.0000000000000004];
    %t=[0.0000000000000000  0.0250000000000000  0.0500000000000000  0.0750000000000000  0.1000000000000000  0.1250000000000000  0.1500000000000000  0.1750000000000000  0.2000000000000000  0.2250000000000000  0.2500000000000000  0.2750000000000000  0.3000000000000000  0.3250000000000000  0.3500000000000000  0.3750000000000000  0.4000000000000000  0.4250000000000000  0.4500000000000000  0.4750000000000000  0.5000000000000000  0.5250000000000000  0.5500000000000000  0.5750000000000000  0.6000000000000000  0.6250000000000000  0.6500000000000000  0.6750000000000000  0.7000000000000000  0.7250000000000000  0.7500000000000000  0.7750000000000000  0.8000000000000000  0.8250000000000000  0.8500000000000000  0.8750000000000000  0.9000000000000000  0.9250000000000000  0.9500000000000000  0.9750000000000000  1.0000000000000000];
    %MIGHT NEED TO LINEARLY RESCALE THE T's  
    
   ti=zeros(1,41); 
   d=zeros(1,41);
   
   %initial t
   d(1)=0;
   for i=1:40
       d(i+1) = d(i)+(sqrt((x(i+1)-x(i))^2+(y(i+1)-y(i))^2));
   end
   
   for i=1:41
   ti(i) = d(i)/d(41);
   end
   
    
    
for u=1:100
    
    m=41;
     syms a1 a2;
     v0 = [x(2)-x(1), y(2)-y(1)];
     v0 = v0/norm(v0);
     v3 = [x(41)-x(40), y(41)-y(40)];
     v3 = v3/norm(v3);
     ex = 0;
     ey = 0;
     P0 = [x(1), y(1)];
     P3 = [x(41), y(41)];
     P1x(a1) = P0(1) + a1*v0(1);
     P1y(a1) = P0(2) + a1*v0(2);
     P2x(a2) = P3(1) + a2*v3(1);
     P2y(a2) = P3(2) + a2*v3(2);
     for i = 1:m;
         ex = ex + (x(i) - (P0(1)*(1-ti(i))^3 + P1x(a1)*3*ti(i)*((1-ti(i))^2) + P2x(a2)*3*(ti(i)^2)*(1-ti(i)) + P3(1)*(ti(i)^3)))^2;
         ey = ey + (y(i) - (P0(2)*(1-ti(i))^3 + P1y(a1)*3*ti(i)*((1-ti(i))^2) + P2y(a2)*3*((ti(i))^2)*(1-ti(i)) + P3(2)*(ti(i)^3)))^2;
     end
     E =(1/m) * (ex + ey);
     dEda1 = diff(E, a1);
     dEda2 = diff(E, a2);
     
     S = vpasolve(dEda1 == 0, dEda2 == 0);
     
     P1 = double([P0(1) + S.a1*v0(1), P0(2) + S.a1*v0(2)]);
     P2 = double([P3(1) + S.a2*v3(1), P3(2) + S.a2*v3(2)]);
     plot(x, y);
     hold on;
     cubicbezier(P0(1), P0(2), P1(1), P1(2), P2(1), P2(2), P3(1), P3(2));
     
     
     
    xB = zeros(1, m);
    yB = zeros(1, m);
    
    
    %bezier
    teller = 1;
    for t = ti
        B0 = (1 - t)^3;
        B1 = 3 * t * (1 - t)^2;
        B2 = 3 * t^2 * (1 - t);
        B3 = t^3;
        
        
        xB(teller) = P0(1)*B0 + P1(1)*B1 + P2(1)*B2 + P3(1)*B3;
        yB(teller) = P0(2)*B0 + P1(2)*B1 + P2(2)*B2 + P3(2)*B3;
        
        teller = teller+1;
        
    end
    
    
    %derivertbezier
        xBd=zeros(1,m);
        yBd=zeros(1,m);
        teller2=1;
        
        for t = ti
            
        B0 = 3*(1 - t)^2;
        B1 = 6 * t * (1 - t);
        B2 = 3 * t^2;
        
        xBd(teller2) = (P1(1)-P0(1))*B0 + (P2(1)-P1(1))*B1 + (P3(1)-P2(1))*B2;
        yBd(teller2) = (P1(2)-P0(2))*B0 + (P2(2)-P1(2))*B1 + (P3(2)-P2(2))*B2;
        
        teller2 = teller2+1;
        
        end
        
        %Finner dobbelderivert avstand i hvert punkt av bezier kurven
    xBdd=zeros(1,m);
    yBdd=zeros(1,m);
    teller3=1;
        
        for t = ti
            
        B0 = 6*(1 - t);
        B1 = 6 * t ;
        
        xBdd(teller3) = (P2(1)-2*P1(1)+P0(1))*B0 + (P3(1)-2*P2(1)+P1(1))*B1;
        yBdd(teller3) = (P2(2)-2*P1(2)+P0(2))*B0 + (P3(2)-2*P2(2)+P1(2))*B1;
        
        teller3 = teller3+1;
        
        end
        
        
    %Finner avstand i hvert punkt av bezier kurven
    %Finner derivert avstand i hvert punkt av bezier kurven
    %dobbelderivert ogs�
    dx=zeros(1,m);
    dy=zeros(1,m);
    ddx=zeros(1,m);
    ddy=zeros(1,m);
    dddx=zeros(1,m);
    dddy=zeros(1,m);
        
    for i = 1:m
    
    dx(i) = ((xB(i)) - x(i))^2;
    dy(i) = ((yB(i)) - y(i))^2;
    
    ddx(i) = 2*(xB(i)-x(i))*xBd(i);
    ddy(i) = 2*(yB(i)-y(i))*yBd(i);
    
    dddx(i)= (xBd(i))^2 + (xB(i)-x(i))*xBdd(i);
    dddy(i)= (yBd(i))^2 + (yB(i)-y(i))*yBdd(i);
    
    end
    
    
    
    
    for i= 2:m-1
        ti(i)= ti(i)-((ddx(i)+ddy(i))/(dddx(i)+dddy(i)));   
    end
    
    disp(sum(dx)+sum(dy));
    
end
    
end