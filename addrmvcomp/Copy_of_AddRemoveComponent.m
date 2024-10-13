function [new_inds,new_inde,new_indr,new_alpha,new_bias] = AddRemoveComponent(X,y,C,inds,inde,indr,alpha,bias,feature,op)

    %   Algorithm to add/remove a component to/from the inner product
    %   Inputs:
    %   - X : data
    %   - Y : labels
    %   - C : regularization parameter
    %   - inds : Margin Support Vectors indices
    %   - inde : Error Support Vectors indices
    %   - indr : Reserve Vectors indices
    %   - alpha : alpha coefficients
    %   - bias : bias term of the model
    %   - feature : the feature to be added or removed (column vector)
    %   - op : 'add' or 'remove' (specify whether to add or remove the feature)
    %   Outputs:
    %   - new_alpha : Updated Lagrange multipliers 
    %   - new_bias : Updated bias term
    
    % Initialization
    [L, N] = size(X); % L : number of data points, N : number of features
    S = X(inds,:);
    E = X(inde,:);
    R = X(indr,:);

    yS = y(inds);
    yE = y(inde);
    fS = feature(inds);
    fE = feature(inde);

    Q = S * S'; % Q : initial linear kernel matrix

    Q_prime = computeQprime(Q,yS);
    U = computeU(yS,fS);
    v = computeV(C,yS,yE,fS,fE);
    
    % Target definition depending on the operation type
    switch op
        case 'add'
            target = 1;
        case 'remove'
            target = -1;
        otherwise
            error('Invalid operation for AddRemoveComponent() function. Use "add" or "remove".')
    end

    % Main loop: Incrementally update the SVM solution
    new_alpha = alpha;
    new_bias = bias;

    old_inds = inds;
    old_inde = inde;
    old_indr = indr;

    p = 0;
    while abs(p) < 1
        
        [delta_p,~,StoR,StoE,EtoS,RtoS]=min_deltap(X,y,feature,C,new_alpha,new_bias,old_inds,old_indr,old_inde,inv(Q_prime),p,target);
        
        [delta_alphaS, delta_bias] = solve_equation(Q_prime,U,v,new_alpha(old_inds),delta_p);

        Q_prime = Q_prime + delta_p * U; % new_Q' = Q' (+/-) delta_p * U
        new_alpha(old_inds) = new_alpha(old_inds) + delta_alphaS; % updated alpha
        new_bias = new_bias + delta_bias; % updated bias

        new_inds = old_inds;
        new_inde = old_inde;
        new_indr = old_indr;
        
        M = 0;
        StoR = find(StoR);
        if ~isempty(StoR)
            new_indr = [new_indr;StoR];
            indices = ismember(new_inds, StoR);
            new_inds(indices) = [];
            M = -1;
            new_alpha(StoR) = 0;
        end
        StoE = find(StoE);
        if ~isempty(StoE)
            new_inde = [new_inde;StoE];
            indices = ismember(new_inds, StoE);
            new_inds(indices) = [];
            M = -1;
            new_alpha(StoE) = C;
        end
        EtoS = find(EtoS);
        if ~isempty(EtoS)
            new_inds = [new_inds;EtoS];
            indices = ismember(new_inde, EtoS);
            new_inde(indices) = [];
            M = 1;
        end
        RtoS = find(RtoS);
        if ~isempty(RtoS)
            new_inds = [new_inds;RtoS];
            indices = ismember(new_indr, RtoS);
            new_indr(indices) = [];
            M = 1;
        end

        new_inds = sort(new_inds);
        new_inde = sort(new_inde);
        new_indr = sort(new_indr);

        if (M ~= 0)
            S = X(new_inds,:);
            E = X(new_inde,:);

            yS = y(new_inds);
            yE = y(new_inde);
            fS = feature(new_inds);
            fE = feature(new_inde);

            if M == -1
                disp('Removing example(s) from S...');
                aux = setdiff(old_inds,new_inds);
                removed_inds = [];
                for i = aux'
                    removed_inds = [removed_inds;find(old_inds == i)];
                end
                Q_prime = contractQ(inv(Q_prime),removed_inds);

                % Update U removing indices
                U(removed_inds,:) = [];
                U(:,removed_inds) = [];
                
                % Update v removing indices
                v(removed_inds) = [];
            elseif M == 1
                disp('Adding example(s) to S...');
                Q_prime = expandQ(X,y,inv(Q_prime),feature,old_inds,new_inds,p+delta_p);
                
                U = computeU(yS,fS);
                v = computeV(C,yS,yE,fS,fE);
            end
        end
        p = p + delta_p;
        old_inds = sort(new_inds);
        old_inde = sort(new_inde);
        old_indr = sort(new_indr);
    end
    new_inds = old_inds;
    new_inde = old_inde;
    new_indr = old_indr;
end