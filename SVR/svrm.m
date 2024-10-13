function [svmSate] = svrm(X,Y,C,e,ker,ktype,sigma)
    
    global KTYPE KSCALE p1
    global verbose debug
    
    KTYPE = ktype;
    if ktype == 6
        KSCALE = (1/(sigma*2))^2;
    else
        KSCALE = 1;
    end

    verbose = 0;           
    comparacio=0;
    debug=0;
    
    % Normalize data
    isotropic = 0;
    [X, A, B] = svdatanorm(X,ker,isotropic);
    X=X(1:end); 
    isotropic=1;
    MY=max(Y);
    mY=min(Y);
    Y=Y/(MY-mY);
    Y=Y(1:end);

    [Lx,tx]= size(X);
    indtrn= [1 Lx];
    
    mag = 0.0001;
    xaxis = 1;
    
    trnX = X;
    trnY = Y;
    tstX = [];
    tstY = [];
    
    % Incremental addition of points
    fprintf('\n  ** Incremental addition of %g points **\n\n',length(trnX));
    
    % Initialize SVM state strcuture
    svmState = struct();
    svmState.a = [];
    svmState.b = 0;
    svmState.g = [];
    svmState.inds = [];
    svmState.indss = [];
    svmState.inde = [];
    svmState.indes = [];
    svmState.support = [];
    svmState.ing = [];
    svmState.ings = [];
    svmState.indo = [];
    svmState.R = [];
    svmState.Qs = [];
    svmState.Qc = [];
    svmState.processed = [];
    svmState.eps = [];
    svmState.eps2 = [];
    svmState.tol = [];

    st = cputime;
    trnX1=trnX(1);
    trnY1=trnY(1);
    svmState = svrm_train(trnX1, trnY1, C, e);  % First data
    for trnexmp=2:length(trnX)
      trnX2=trnX(trnexmp); trnY2=trnY(trnexmp);
      [svmState, error] = svrm_train_afegir(svmState, trnX1, trnY1, trnX2, trnY2, C, e);
      trnX1=[trnX1; trnX2]; trnY1=[trnY1; trnY2];
    end
    fprintf('Execution time : %4.1f seconds\n',cputime - st)
    
    ls=length(svmState.inds)+length(svmState.indss);
    le=length(svmState.inde)+length(svmState.indes);
    [Lx,tx]=size(X);
    fprintf('\nSupport vectors : %g (%4.1f)', ls, ls*100/Lx);
    fprintf('\nError vectors   : %g (%4.1f)\n\n', le, le*100/Lx);
end