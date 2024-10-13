function [U] = computeU(yS,fS)
    
    %   Method to compute Q' matrix
    %   Inputs:
    %   - yS : labels for Margin Support Vectors
    %   - fS : column vector containing new feature values for S
    %   Outputs:
    %   - U  : transformation matrix L_S x L_S
    
    L_S = size(fS,1);
    U = zeros(L_S+1,L_S+1);
    for i = 1:L_S
        for j = 1:L_S
            U(i+1,j+1) = yS(i) * yS(j) * dot(fS(i),fS(j));
        end
    end

end