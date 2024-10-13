function [Q_prime] = computeQprime(Q,yS)
    
    %   Method to compute Q' matrix
    %   Inputs:
    %   - Q  : matrix with the inner products for Margin Support Vectors
    %   - yS : labels for Margin Support Vectors
    %   Outputs:
    %   - Q' : prime kernel matrix L_S+1 x L_S+1
    
    L_S = size(Q, 1);
    Q_prime = zeros(L_S+1,L_S+1);
    for i = 1:L_S
        for j = 1:L_S
            Q_prime(i+1,j+1) = yS(i) * yS(j) * Q(i,j);
        end
    end

    Q_prime(2:end,1) = yS;
    Q_prime(1,2:end) = yS';
end