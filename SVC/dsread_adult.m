function [X,Y] = dsread_adult()
    warning off;
    data = readtable("adult.csv");

    X = zeros(height(data),width(data)-1);
    Y = zeros(length(X),1);
    
    for i = 1:width(data)-1
        
        if iscell(data{:, i})
            % Convert the cell array to categorical, then to double
            X(:, i) = double(categorical(data{:, i}));
        else
            X(:,i) = data{:,i};
        end
    end
    
    Y = double(categorical(data{:, end}));
    Y(Y == 2) = -1;
    warning on;
end