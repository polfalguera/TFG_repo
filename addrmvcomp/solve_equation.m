function [delta_alphaS, delta_bias] = solve_equation(Q_prime,U,v,alphaS,delta_p)
    
    %   Method to solve delta_alpha and delta_bias
    %   Inputs:
    %   - Q_prime: The matrix Q'
    %   - U: The matrix U
    %   - alpha_S: The vector [alpha_S1; ...; alpha_S_{L_S}]
    %   - v: The vector v
    %   - delta_p: The scalar Δp
    %   Outputs:
    %   - U  : transformation matrix L_S x L_S

    %   Inputs:
    %   - Q_prime: The matrix Q'
    %   - U: The matrix U
    %   - alpha_S: The vector [alpha_S1; ...; alpha_S_{L_S}]
    %   - v: The vector v
    %   - delta_p: The scalar Δp
    
    % Form the left-hand side matrix
    try
        lhs_matrix = (Q_prime / delta_p) + U;
    catch
        disp('ERROR')
    end
    
    % Form the right-hand side vector
    rhs_vector = -U * [0;alphaS] - v;
    
    % Solve for [Δb; Δα_S1; ...; Δα_S_{L_S}]
    delta_vars = lhs_matrix \ rhs_vector; % Using MATLAB's backslash operator for solving linear systems

    delta_alphaS = delta_vars(2:end);
    delta_bias = delta_vars(1);

end