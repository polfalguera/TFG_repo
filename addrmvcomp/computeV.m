function [v] = computeV(C,yS,yE,fS,fE)

    %   Method to compute Q' matrix
    %   Inputs:
    %   - E : Error Support Vectors
    %   - C : regularization parameter
    %   - yS : labels for Margin Support Vectors
    %   - yE : labels for Error Support Vectors
    %   - fS : column vector containing new feature values for S
    %   - fE : column vector containing new feature values for E
    %   Outputs:
    %   - v  : column vector of L_S+1 x 1

    L_S = size(fS,1);
    v = zeros(L_S+1, 1);

    dprod = dot(yE,fE);

    for i = 1:L_S
        v(i+1) = C * yS(i) * fS(i) * dprod;
    end

end