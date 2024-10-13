function SnapPath(step, x, y, f, alpha, alpha0, lambda, Elbow, kernel_function, param_kernel, linear_plot, Size, movie, movie_root, varargin)
    if nargin < 13
        movie = false;
        movie_root = './';
    end
    if nargin < 12
        Size = 60;
    end

    if movie
        filename = sprintf('%s%d.jpg', movie_root, step);
        f = figure('visible', 'off');
        set(f, 'Position', [100, 100, 540, 540], 'Color', 'wheat');
    end

    stats = StatPath(y, f, Elbow);

    %%% Only for 2 dim inputs

    x = x(:, 1:2);
    n = length(y);
    ss = abs(range(x(:, 2)));
    soffset = zeros(size(x));
    soffset(:, 2) = ss / 50;

    if ~linear_plot
        xr = [min(x); max(x)];
        xg = arrayfun(@(i) linspace(xr(1, i), xr(2, i), Size), 1:2, 'UniformOutput', false);
        [X1, X2] = ndgrid(xg{:});
        xG = [X1(:), X2(:)];
        Knew = kernel_function(x, xG, param_kernel, varargin{:});
        fhat = ((alpha .* y') * Knew + alpha0) / lambda;
        fhat = reshape(fhat, Size, Size);
    end

    hold on;
    axis equal;
    xlabel('X1');
    ylabel('X2');
    title(sprintf('              Step: %d   Error: %.3f   Elbow Size: %.2f   Sum Eps: %.7f', ...
        step, round(stats.error), round(stats.selbow, 2), round(stats.margin, 2)), 'HorizontalAlignment', 'left');
    
    pointid = 1:n;
    scatter(x(y == 1, 1), x(y == 1, 2), 100, 'r', '.');
    scatter(x(y == -1, 1), x(y == -1, 2), 100, 'b', '.');
    
    if n < 15
        text((x(y == 1, 1) - soffset(y == 1, 1)), (x(y == 1, 2) - soffset(y == 1, 2)), arrayfun(@num2str, pointid(y == 1), 'UniformOutput', false), 'Color', 'r');
        text((x(y == -1, 1) - soffset(y == -1, 1)), (x(y == -1, 2) - soffset(y == -1, 2)), arrayfun(@num2str, pointid(y == -1), 'UniformOutput', false), 'Color', 'b');
    end
    
    if n < 15 && ~isempty(Elbow)
        text((x(Elbow, 1) - soffset(Elbow, 1)), (x(Elbow, 2) - soffset(Elbow, 2)), arrayfun(@num2str, pointid(Elbow), 'UniformOutput', false), 'Color', 'g');
    end
    if linear_plot
        beta = (alpha .* y') * x;
        beta = beta(1,:);
        y_intercept = -alpha0 / beta(2);
        slope = -beta(1) / beta(2);
        line_x = linspace(min(x(:, 1)), max(x(:, 1)), 100);
        line_y = slope * line_x + y_intercept;
        plot(line_x, line_y, 'g', 'LineWidth', 3);

        
        line_y = slope * line_x + (lambda - alpha0) / beta(2);
        plot(line_x, line_y, 'g--', 'LineWidth', 2);
        
        line_y = slope * line_x + (-lambda - alpha0) / beta(2);
        plot(line_x, line_y, 'g--', 'LineWidth', 2);
    else
        contour(X1, X2, fhat, [0, 0], 'Color', 'g', 'LineWidth', 3);
        contour(X1, X2, fhat, [-1, 1], 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
    end
    
    if movie
        saveas(f, filename);
        close(f);
    else
        hold off;
    end
end