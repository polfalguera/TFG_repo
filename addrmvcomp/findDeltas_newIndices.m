function [res] = findDeltas_newIndices(X,y,C,Q_prime,U,v,new_alpha,new_bias,old_inds,old_inde,old_indr,feature,p,target)
    ext = target - p;
    inc = target * 0.1;
    interval = inc:inc:ext;

    alphaS = new_alpha(old_inds);

    res = struct();
    try
        res.delta_p = interval(1);
    catch
        res.delta_p = inc;
    end
    [res.delta_alphaS, res.delta_bias] = solve_equation(Q_prime,U,v,alphaS,p+res.delta_p);
    res.alpha = new_alpha;
    res.alpha(old_inds) = res.alpha(old_inds) + res.delta_alphaS;
    res.alpha(res.alpha < 0) = 0;
    res.bias = new_bias + res.delta_bias;
    [res.inds,res.inde,res.indr,res.M] = checkUpdateSets(X,y,C,res.alpha,res.bias,old_inds,old_inde,old_indr,feature,p+res.delta_p);
    
    for i = interval(2:end)
        curr = struct();
        curr.delta_p = i;
        [curr.delta_alphaS, curr.delta_bias] = solve_equation(Q_prime,U,v,alphaS,curr.delta_p);
        curr.alpha = new_alpha;
        curr.alpha(old_inds) = curr.alpha(old_inds) + curr.delta_alphaS;
        curr.alpha(curr.alpha < 0) = 0;
        curr.bias = new_bias + curr.delta_bias;
        [curr.inds,curr.inde,curr.indr,curr.M] = checkUpdateSets(X,y,C,curr.alpha,curr.bias,old_inds,old_inde,old_indr,feature,p+curr.delta_p);
        if (curr.delta_p > res.delta_p) && (curr.M <= res.M)
            res = curr;
        end
        % fprintf('For delta_p %d the given M = %d\n', curr.delta_p, curr.M);
    end
end