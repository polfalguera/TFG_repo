function obj = svmpath(x,y,kernel_function,param_kernel,trace,plot_it,eps,Nmoves,digits,lambda_min,ridge) % removed K
if nargin < 11
    ridge = 0;
end
if nargin < 10
    lambda_min = 1e-4;
end
if nargin < 9
    digits = 6;
end
if nargin < 8
    Nmoves = 3 * length(y);
end
if nargin < 7
    eps = 1e-10;
end
if nargin < 6
    plot_it = false;
end
if nargin < 5
    trace = false;
end
if nargin < 4
    param_kernel = 1;
end
if nargin < 3
    kernel_function = @poly_kernel;
end

K = kernel_function(x,param_kernel);

linear_plot = isequal(kernel_function, @poly_kernel) && param_kernel == 1;

if plot_it && size(x,2) > 2
    error('Plotting only for 2-dim X');
end

n = length(y);
yvals = tabulate(y);

if size(yvals,1) ~= 2
    error('SvmPath works with binary problems only');
end

nplus = yvals(2,2);
nminus = yvals(1,2);

Right = [];
Elbow = [];
Left = 1:n;
Kscript = K .* (y*y');

%%% We start with a maximum of 2*n moves,but these can be increased
%%%
%%% Initializations of counters

alpha = ones(n,Nmoves);
alpha0 = zeros(1,Nmoves);
SumEps = zeros(1,Nmoves);
Elbow_list = cell(1,Nmoves);
Size_Elbow = zeros(1,Nmoves);
Error = zeros(1,Nmoves);
Step = zeros(1,2 * Nmoves);
Obs_step = zeros(1,2 * Nmoves);
Movefrom = strings(1,2 * Nmoves);
Moveto = strings(1,2 * Nmoves);
lambda = zeros(1,Nmoves);
Kstar = zeros(1);

%%% Initialization of path
%%% Two cases nplus=nminus or else not
if nplus == nminus
    init = BalancedInitialization(K,y,nplus,nminus);
    Elbow = init.Elbow; 
    Left = setdiff(Left,Elbow);
else
    init = UnbalancedInitialization(K,y,nplus,nminus);
    Elbow = init.Elbow;
    Right = init.Right;
    Left = init.Left;
end

Elbow_list{1} = Elbow;
lambda0 = init.lambda;
alpha0(1) = init.alpha0;
alpha(:,1) = init.alpha;
alpha00 = init.alpha00; % safekeeping
Kstar = UpdateKstar(Kstar,Kscript(Elbow,Elbow),[],y(Elbow)); 
lambda(1) = lambda0;
fl = (K * (alpha(:,1) .* y) + init.alpha0) / lambda0;
stats = StatPath(y,fl,Elbow);
SumEps(1) = stats.margin;
Error(1) = stats.error;
Size_Elbow(1) = stats.selbow;
nobs = 1:length(Elbow);
Step(nobs) = 1;
Obs_step(nobs) = Elbow;
Movefrom(nobs) = " ";
Moveto(nobs) = "E";
move_counter = length(nobs);
PrintPath(trace,1,Elbow,"E"," ",lambda0,digits,stats);
k = 1;

if plot_it
    SnapPath(k,x,y,fl,alpha(:,k),alpha0(k),lambda(k),Elbow,kernel_function,param_kernel,linear_plot);
end

while (k < Nmoves) && (lambda(k) > lambda_min)
    %%% Now we implement the updates in Section 4.0
    if isempty(Elbow)
        %%% The elbow has become empty; need to resort to an initial condition
        if sum(y(Left)) ~= 0
            error('Unbalanced data in interior empty elbow situation');
        end
        init = BalancedInitialization(K(Left,Left),y(Left),length(Left) / 2);
        lambda0 = init.lambda;
        alpha0(k + 1) = init.alpha0;
        Elbow = Left(init.Elbow);
        Left = setdiff(Left,Elbow);
        lambda(k + 1) = lambda0;
        alpha(:,k + 1) = alpha(:,k);
        Kstar = UpdateKstar(Kstar,Kscript(Elbow,Elbow),[],y(Elbow));
        fl = (lambda(k) / lambda(k + 1)) * (fl + (alpha0(k + 1) - alpha0(k)) / lambda(k));
        movefrom = " ";
        moveto = "E";
        obs = Elbow;
    else
        bstar = SolveKstar(Kstar,ridge);
        b0 = bstar(1);
        b = bstar(2:end);

        %%% Now find the first event
        %%% Check for immobil margin

        gl = K(:,Elbow) * (y(Elbow)' .* b)' + b0;
        dl = fl - gl;
        immobile = any(sum(abs(dl)) / n < eps);
        
        %%% Now check for exits from Elbow

        temp = -alpha(Elbow,k)' + lambda(k) * b;
        lambda_left = (1 + temp) ./ b;
        lambda_left(abs(b) < eps) = -1;  % anything negative
        lambda_right = temp ./ b;
        lambda_right(abs(b) < eps) = -1;
        lambda01 = [lambda_right,lambda_left];
        lambda_exit = max(lambda01(lambda01 < lambda(k) - eps));

        %%% Check to see if we leave the margin when it is immobile
        
        if immobile && (lambda_exit  < eps)
            break;
        end

        %%% Now check for entries
        
        if ~immobile
            lambdai = (lambda(k) * dl) ./ (y - gl);
            lambdai(abs(y - gl) < eps) = -Inf;
            lambda_entry = max(lambdai(lambdai < lambda(k) - eps));
        else
            lambda_entry = -1; % any negative will do
        end

        lambda_max = max(lambda_entry,lambda_exit);

        %%% Update lambda, alphas, fit
    
        if (k == 79)
            disp('hello')
        end
        lambda(k + 1) = lambda_max;
        alpha(:, k + 1) = alpha(:, k);
        alpha(Elbow, k + 1) = (alpha(Elbow, k)' - (lambda(k) - lambda_max) .* b)';
        alpha0(k + 1) = alpha0(k) - (lambda(k) - lambda(k + 1)) * b0;
        fl = (lambda(k) / lambda(k + 1)) .* dl + gl;

        %%% Update active sets

        if (lambda_entry > lambda_exit)
            
            %%% Point joins the elbow
            [~,i] = max(lambda_entry == lambdai);
            obs = i;

            %%% Assumes for now there is only 1

            moveto = "E";
            if ismember(i,Left)
                Left = setdiff(Left,i);
                movefrom = "L";
            else
                Right = setdiff(Right,i);
                movefrom = "R";
            end
            Kstar = UpdateKstar(Kstar, Kscript(i, i), Kscript(i, Elbow), y(i));
            Elbow = [Elbow,i];
        else

            %%% Point(s) leaves the elbow; can be more than one

            movefrom = "E";
            moveto = [];
            idrop = [];
            Leaveright = [];
            
            i = Elbow(abs(lambda_right - lambda_exit) < eps);
            if ~isempty(i)
                Leaveright = true(length(i));
                idrop = i;
            end

            i = Elbow(abs(lambda_left - lambda_exit) < eps);
            if ~isempty(i)
                Leaveright = false(length(i));
                idrop = [idrop,i];
            end

            obs = idrop;
            for j = 1:length(idrop)
                if Leaveright(j)
                    moveto = [moveto,"R"];
                    Right = [Right,idrop(j)];
                else
                    moveto = [moveto,"L"];
                    Left = [Left,idrop(j)];
                end

                mi = find(idrop(j) == Elbow);
                Kstar = DowndateKstar(Kstar,mi);
                Elbow(mi) = [];
            end

        end

    end

    k = k + 1;
    stats = StatPath(y, fl, Elbow);
    SumEps(k) = stats.margin;
    Error(k) = stats.error;
    Size_Elbow(k) = stats.selbow;
    nobs = 1:length(obs);
    Moveto(move_counter + nobs) = moveto;
    Movefrom(move_counter + nobs) = movefrom;
    Step(move_counter + nobs) = k;
    Obs_step(move_counter + nobs) = obs;
    move_counter = move_counter + length(nobs);
    Elbow_list{k} = Elbow;
    PrintPath(trace, k, obs, moveto, movefrom, lambda(k), digits, stats);
    if plot_it
        SnapPath(k, x, y, fl, alpha(:, k), alpha0(k), lambda(k), Elbow, kernel_function, param_kernel, linear_plot);
    end
    fprintf('- Loop %d completed.\n', k);
end
obj = struct('alpha', alpha(:, 1:k), 'alpha0', alpha0(1:k), 'lambda', lambda(1:k), ...
        'alpha00', alpha00, 'Error', Error(1:k), 'SumEps', SumEps(1:k), ...
        'Size_Elbow', Size_Elbow(1:k), 'Elbow', {Elbow_list(1:k)}, 'Moveto', {Moveto(1:move_counter)}, ...
        'Movefrom', {Movefrom(1:move_counter)}, 'Obs_step', Obs_step(1:move_counter), 'Step', Step(1:move_counter), ...
        'kernel', kernel_function, 'param_kernel', param_kernel, 'x', x, 'y', y, 'linear', linear_plot);
end

