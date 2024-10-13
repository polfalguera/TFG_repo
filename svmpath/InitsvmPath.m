function result = InitsvmPath(Rmat, cvec, const)

    n = length(cvec);
    Aeq = ones(1, n);
    beq = const;
    lb = zeros(n, 1);
    ub = ones(n, 1);
    
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'TolFun', 1e-5, 'Display', 'off');

    [alpha, ~, exitflag] = quadprog(Rmat, cvec, [], [], Aeq, beq, lb, ub, [], options);
    
    if exitflag ~= 1
        error('Optimization did not converge');
    end
    
    result.alpha = alpha;
end
