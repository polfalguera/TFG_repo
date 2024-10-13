function [new_inds,new_inde,new_indr,M] = checkUpdateSets(X,y,C,new_alpha,new_bias,old_inds,old_inde,old_indr,feature,p)
    
    new_inds = old_inds;
    new_inde = old_inde;
    new_indr = old_indr;

    M = 0; % Total number of migrations

    % Check for migrations from S to E,R
    for i = old_inds'
        switch new_alpha(i)
            case 0 
                M = M + 1;
                new_inds(new_inds == i) = [];
                new_indr = [new_indr;i];
            case C
                M = M + 1;
                new_inds(new_inds == i) = [];
                new_inde = [new_inde;i];
        end
    end

    % Check for migrations from E to S
    for i = old_inde'
        try
            g_i = computeG(X,y,new_alpha,new_bias,feature,p,i);

            if g_i == 0
                M = M + 1;
                new_inde(new_inde == i) = [];
                new_inds = [new_ind;i];
            end
        catch
            disp('NOTHING TO CHECK');
        end
    end

    % Check for migrations from R to S
    for i = old_indr'
        try
            g_i = computeG(X,y,new_alpha,new_bias,feature,p,i);
            
            if i == 24
                disp('nothing')
            end
            if g_i == 0
                M = M + 1;
                new_indr(new_indr == i) = [];
                new_inds = [new_inds;i];
            end
        catch
            disp('NOTHING TO CHECK');
        end
    end

end