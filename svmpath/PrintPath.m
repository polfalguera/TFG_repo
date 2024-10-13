function PrintPath(trace, step, obs, moveto, movefrom, lambda, digits, stats)
    if trace
        moveto = repmat(moveto, 1, length(obs));
        movefrom = repmat(movefrom, 1, length(obs));
        for i = 1:length(obs)
            fprintf('%d:\tObs %d\t%s->%s  lambda = %.*f  Sum Eps = %.2f  Elbow = %d  Error = %d\n', ...
                step, obs(i), movefrom(i), moveto(i), digits, lambda, round(stats.margin, 2), stats.selbow, stats.error);
        end
    end
end