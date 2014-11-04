function task2(filename)
    file = strcat(filename, '.txt');
    
    fileID = fopen(file, 'r');
    
    [Pvector, count] = fscanf(fileID, ['%d' '%d' '%d' '%d' '%d' '%d' '%d' '%d' ';']);
    
    Points = zeros((count/8), 8);
    k = 1;
    
    for i = 1 : count/8
        for j = 1 : 8
            Points(i, j) = Pvector(k);
            k = k + 1;
        end
    end
    
    for i = 1 : length(Points(:, 1))
        
    end

    


end

