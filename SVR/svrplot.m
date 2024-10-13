function svrplot(X, Y, ker, beta, bias, e, mag, xaxis, Ax, Bx, mY, MY)
% SVRPLOT Support Vector Regression Plotting routine for 1D function
%
%  Usage: svrplot(X, Y, ker, alpha, bias, e, mag, xaxis, yaxis)
%
%  Parameters:
%    X      - Training inputs
%    Y      - Training targets
%    ker    - Kernel function
%    beta   - Difference of Lagrange Multipliers
%    bias   - Bias term 
%    e      - e insensitive value
%    mag    - Display magnification 
%    xaxis  - xaxis input (default: 1) 
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)

    narginchk(5, 12); % Check correct number of arguments

    if nargin < 12, mY = 0; MY = 1; end
    if nargin < 10, Ax = 1; Bx = 0; end 
    if nargin < 9, xaxis = 1; end
    if nargin < 8, mag = 0; end
    if nargin < 7, e = 0; end

    epsilon = 1e-4;  

    xmin = min(X(:, xaxis)); 
    xmax = max(X(:, xaxis)); 
    xa = xmax - xmin;
    xmin = xmin - mag * xa;
    xmax = xmax + mag * xa;

    % Generate x values for the plot
    x = linspace(xmin, xmax, 300); 
    z = bias * ones(size(x));

    % Compute the function values
    for i = 1:length(beta)
        if abs(beta(i)) > epsilon
            z = z + beta(i) * arrayfun(@(xi) svkernel(ker, setInputValue(xi, xaxis, X(i, :)), X(i, :)), x);
        end
    end

    % Scale the x and z values
    x = (x - Bx) / Ax;
    z = z * (MY - mY);
    e = e * (MY - mY);

    % Plot the function
    plot(x, z, 'k', 'LineWidth', 2);
    hold on

    % Plot e-insensitive zone
    if e > 0
        plot(x, z + e, 'b:');
        plot(x, z - e, 'b:');
    end

    % Plot training points and support vectors
    X = (X - Bx) / Ax;
    Y = Y * (MY - mY);
    plot(X, Y, 'k+');
    supportVectors = abs(beta) > epsilon;
    plot(X(supportVectors), Y(supportVectors), 'ro', 'MarkerSize', 10);

    % Set axis limits
    set(gca, 'XLim', ([xmin xmax] - Bx) / Ax);  
    hold off
end

function input = setInputValue(value, xaxis, inputVector)
    % Helper function to set the xaxis value in the input vector
    inputVector(xaxis) = value;
    input = inputVector;
end
