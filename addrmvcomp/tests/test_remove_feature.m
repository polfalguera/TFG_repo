clear;
load fisheriris

X1 = meas(1:100,:);
y1 = [ones(50,1);repmat(-1,50,1)];

X2 = meas(1:100,1:3);
aux = meas(1:100,end);

%% SECOND DATASET

clear;
load cancer_dataset

X1 = cancerInputs';
y1 = vec2ind(cancerTargets)';
y1 = 2 * (y1 - 1) - 1;

X2 = X1(:, 1:end-1);
aux = X1(:,end);

clear cancerInputs cancerTargets;

%% TRAIN M1

m1 = fitcsvm(X1,y1,"BoxConstraint",100,"KernelFunction","linear", "Solver","L1QP");
inds1 = find(m1.IsSupportVector);

% OBTAIN PARAMETERS

alphaFull = zeros(size(X1,1), 1);
alphaFull(m1.IsSupportVector) = m1.Alpha;
C = m1.BoxConstraints(1);
bias = m1.Bias;


inds = find(m1.IsSupportVector);
inde = inds(alphaFull(inds) == C);
inds(ismember(inds,inde)) = [];
indr = find(~m1.IsSupportVector);

% TRAIN M2

m2 = fitcsvm(X2,y1,"BoxConstraint",100,"KernelFunction","linear","Solver","L1QP");
inds2 = find(m2.IsSupportVector);

% REMOVE TEST
[new_inds,new_inde,new_indr,new_alphas,new_bias] = AddRemoveComponent(X1,y1,C,inds,inde,indr,alphaFull,bias,aux,'remove');