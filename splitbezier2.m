function splitbezier2(x, y, v0, v3)
% x = linspace(0,1,101);
% y = sin(pi*x);
m = length(x);
d = zeros(1, m);
for i=1:m-1
   d(i+1) = d(i)+(sqrt((x(i+1)-x(i))^2+(y(i+1)-y(i))^2));
end
ti = d/d(m);

P0 = [x(1), y(1)];
P3 = [x(m), y(m)];
if v0 == 0;
    v0 = [x(2)-x(1), y(2)-y(1)];
    v0 = v0/norm(v0);
end
if v3==0;
    v3 = [x(m-1)-x(m), y(m-1)-y(m)];
    v3 = v3/norm(v3);
end

teller = 1;
A1 = 0;
A11 = 0;
A12 = 0;
A22 = 0;
A2 = 0;
for t = ti;
    b0 = (1-t)^3;
    b1 = 3*t*((1-t)^2);
    b2 = 3*(t^2)*(1-t);
    b3 = t^3;
 
    A = [x(teller), y(teller)] - P0*b0 - P0*b1 - P3*b2 - P3*b3;
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
    if E > 0.00000001;
        v20 = [x(floor(m/2)-1)-x(floor(m/2)), y(floor(m/2)-1)-y(floor(m/2))];
        v20 = v20/norm(v20);
        v21 = [x(floor(m/2)+1)-x(floor(m/2)), y(floor(m/2)+1)-y(floor(m/2))];
        v21 = v21/norm(v21);
        v2 = v20-v21;
        v2 = v2/norm(v2);
        splitbezier2(x(1:floor(m/2)), y(1:floor(m/2)), v0, -v2);
        splitbezier2(x(floor(m/2):m), y(floor(m/2):m), v2, v3);
    else
        disp(E);
        plot(x,y);
        hold on;
        cubicbezier(P0(1), P0(2), P1(1), P1(2), P2(1), P2(2), P3(1), P3(2));
    end
end