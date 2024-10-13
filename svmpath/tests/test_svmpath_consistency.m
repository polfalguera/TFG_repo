
clear; clc;

X = ["X_balanced_overlap.txt", "X_balanced_separated.txt", "X_mixture_data.txt", "X_unbalanced_separated.txt"]; 
Y = ["Y_balanced_overlap.txt", "Y_balanced_separated.txt", "Y_mixture_data.txt", "Y_unbalanced_separated.txt"];
n_datasets = length(X);

lambdas = cell(1,n_datasets);

for idx = 1:n_datasets

    X_train = importdata(X(idx));
    Y_train = importdata(Y(idx));

    res = svmpath(X_train,Y_train,@radial_kernel);
    clc;

    lambdas{idx} = res.lambda;
end