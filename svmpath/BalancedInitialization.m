function [result] = BalancedInitialization(K,y,nplus,nminus,eps)
    if nargin < 4
        nminus = nplus;
    end
    if nargin < 5
        eps = 1e-12;
    end

    n = 2 * nplus;
    f = K * y;
    Iplus = find(y > 0)';
    Iminus = find(y < 0)';
    fmax = max(f(Iplus));
    fmin = min(f(Iminus));
    iplus = Iplus(find(f(Iplus) == fmax, 1));
    iminus = Iminus(find(f(Iminus) == fmin, 1));

    %%% This seems to take into account ties

    lambda = (fmax - fmin) / 2;
    beta0 = 1 - fmax / lambda;
    alpha0 = beta0 * lambda;

    %%% Package parameters for the left of the start

    alpha00 = struct('slope',beta0,'intercept',0);
    
    if (length(iplus) > 1)
        disp('Stop!');
    end
    result.Elbow = [iplus, iminus];
    result.lambda = lambda;
    result.alpha0 = alpha0;
    result.alpha00 = alpha00;
    result.alpha = ones(n, 1);
end