function K = radial_kernel(x, param_kernel)
    % If param_kernel is not provided, default to 1/p (equivalent to 1/ncol(x) in R)
    if nargin < 2
        param_kernel = 1 / size(x, 2);
    end
    
    % Get dimensions
    n = size(x, 1);  % number of rows in x
    m = size(x, 1);  % number of rows in y
    p = size(x, 2);  % number of columns in x (dimensionality)
    
    % Compute norms
    normx = sum(x.^2, 2);  % sum of squares of each row of x
    normy = sum(x.^2, 2);  % sum of squares of each row of y
    
    % Compute the Gram matrix
    a = x * x';            % dot products between rows of x and rows of y
    a = (-2 * a + normx) + repmat(normy', n, 1);
    
    % Compute the RBF kernel matrix
    K = exp(-a * param_kernel);
end
