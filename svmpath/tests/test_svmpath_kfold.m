clear X Y; clc;

[X, Y] = dsread_adult();

X_train = X(1:2000,:);
Y_train = Y(1:2000);

% COMPUTE regular K-fold cross-validation

C_values = [0.01,0.1,1,10,100,1000]; 
k = 10; 
accuracy_kfold = zeros(length(C_values), k);
error_kfold = zeros(length(C_values), k); 

for i = 1:length(C_values)
    C = C_values(i);
    
    cv = cvpartition(Y, 'KFold', k);
    
    for j = 1:k
        X_train = X(training(cv, j), :);
        Y_train = Y(training(cv, j));
        X_val = X(test(cv, j), :);
        Y_val = Y(test(cv, j));
        
        SVMModel = fitcsvm(X_train, Y_train, 'BoxConstraint', C, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
        
        predictions = predict(SVMModel, X_val);
        accuracy_kfold(i, j) = mean(predictions == Y_val);
        error_kfold(i, j) = 1 - accuracy_kfold(i, j);
    end
end

error_kfold_mean = mean(error_kfold, 2);
error_kfold_std = std(error_kfold, 0, 2);

% COMPUTE svmpath for entire regularization path

reg_path = svmpath(X_train, Y_train, @poly_kernel, 3);


