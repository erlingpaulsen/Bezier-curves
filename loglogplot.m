function loglogplot(n);
x = linspace(0,1,101);
y = sin(pi*x);
ys = [];
xs=[];
for i = 0:n
    list = splitbezier2(x, y, 0, 0, (ceil(101/(2^i)))+1);
    ys = [ys, sum(list)/length(list)];
    xs = [xs, length(list)];
end
disp(xs);
hold off;
loglog(xs, ys);
end