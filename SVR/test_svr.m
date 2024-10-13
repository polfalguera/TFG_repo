clear; clc; rng(1);

% Parameters for the dataset
n_samples = 3000;  
n_features = 3;   

% Generate random input features
X = rand(n_samples, n_features) * 10 - 5;
x1 = X(:,1);
x2 = X(:,2);

% Define a polynomial relationship for the output (target variable)
Y = 3*x1.^2 + 2*x1.*x2 - 4*x2.^2 + randn(n_samples, 1) * 5;

X_train = X(1:2000,:);
Y_train = Y(1:2000);

X_test = X(2001:end,:);
Y_test = Y(2001:end);

%% TRAIN MATLAB SVR

svrModel = fitrsvm(X_train,Y_train,'BoxConstraint',100,'KernelFunction','polynomial','PolynomialOrder',2,'Epsilon',0.025);

pl = svrModel.predict(X_test);

fprintf('Accuracy (%%): %d\n', (sum(Y_test == pl)/length(pl)) * 100);

%% TRAIN INCREMENTAL/DECREMENTAL SVR

m = svrm(X,Y,100,0.025,'polynomial',2,1);