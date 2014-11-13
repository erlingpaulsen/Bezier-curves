function t = initT(X)
%INITT Summary of this function goes here
%   Detailed explanation goes here

   m = length(X(1, :));
   d = zeros(1,m);

   %initial t
   d(1) = 0;
   for i = 1 : m - 1
       d(i + 1) = d(i) + (sqrt((X(1, i + 1) - X(1, i))^2 + (X(2, i + 1) - X(2, i))^2));
   end
   
    t = d/d(m);
    
end

