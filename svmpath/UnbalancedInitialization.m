function result = UnbalancedInitialization(K,y,nplus,nminus,eps)
    if nargin < 5
        eps = 1e-6;
    end

    function init = pos_init(K,y,nplus,nminus,eps)
        
        %%% Build up Left and dominant class indices,
        
        Iplus = find(y > 0)';
        Iminus = find(y < 0)';
        Left = Iminus;
        Kscript = K .* (y * y');
        Rmat = Kscript(Iplus, Iplus);
        c_objective = Kscript(Iplus, Iminus) * ones(nminus, 1);
        alpha_opt = InitsvmPath(Rmat, c_objective, nminus).alpha;

        alpha = ones(length(y), 1)';
        alpha(Iplus) = alpha_opt;

        zeros = alpha_opt < eps;
        if any(zeros)
            Right = Iplus(zeros);
        else
            Right = [];
        end

        elbows = (~zeros) & (alpha_opt < 1 - eps);
        if any(elbows)
            Elbow = Iplus(elbows);
        else
            Elbow = [];
        end

        Iplus_Left = setdiff(Iplus, [Right, Elbow]);
        Left = [Left, Iplus_Left];

        f = ((y' .* alpha) * K)';

        fmin = min(f(Iminus));
        
        if ~isempty(Elbow)
            fmax = max(f(Elbow));
        else
            fmax = max(f(Iplus_Left));
            Elbow = Iplus_Left(abs(f(Iplus_Left) - fmax) < eps);
            Left = setdiff(Left, Elbow);
        end

        iminus = Iminus(abs(f(Iminus) - fmin) < eps);
        Elbow = [Elbow, iminus];
        Left = setdiff(Left, iminus);
        lambda = (fmax - fmin) / 2;
        beta0 = 1 - fmax / lambda;
        
        init.Elbow = Elbow;
        init.Right = Right;
        init.Left = Left;
        init.lambda = lambda;
        init.alpha = alpha;
        init.beta0 = beta0;
    end

    %%% Find out wich class is dominant

    if nminus > nplus
        more_y = -1;
        
        %%% We like to work with + class dominant

        init = pos_init(K, -y, nminus, nplus, eps);

        %%% Fix up some of the entries

        init.beta0 = -init.beta0;
    else
        more_y = 1;
        init = pos_init(K, y, nplus, nminus, eps);
    end

    alpha0 = init.beta0 * init.lambda;
    
    %%% Package parameters for the left of the start
    %%% Alpha0 is linear in lambda, so just solve a system of equations

    alpha00 = struct('slope', more_y, 'intercept', init.lambda * (init.beta0 - more_y));
    result = init;
    result.alpha0 = alpha0;
    result.alpha00 = alpha00;
end