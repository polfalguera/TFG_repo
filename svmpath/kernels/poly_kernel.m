function K = poly_kernel(x, param_kernel)

    % Set default value for param_kernel if not provided
    if nargin < 3 || isempty(param_kernel)
        param_kernel = 1;
    end
    
    % Compute the polynomial kernel
    if param_kernel == 1
        K = x * x';
    else
        K = (x * x' + 1) ^ param_kernel;
    end
    
end