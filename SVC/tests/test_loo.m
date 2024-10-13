clear X Y; clc;

[X, Y] = dsread_adult();

X_train = X(1500:2000,:);
Y_train = Y(1500:2000);

%% LOO MatLab common approach

c = cvpartition(size(X, 1), 'Leaveout');

tic
for i = 1:c.NumObservations

    trainIdx = training(c, i); % Indices for training data
    testIdx = test(c, i);      % Indices for test data (one sample)
     
    svmModel = fitcsvm(X_train,Y_train,'BoxConstraint',100,'KernelFunction','rbf','KernelScale',"auto",'Solver','L1QP');
    
    predicted_labels(testIdx) =svmModel.predict(X(testIdx, :));
end
time1 = toc;

LOO_error = sum(predicted_labels ~= Y) / length(Y);

fprintf('Standard LOO error rate : %.4f\n', LOO_error);
fprintf('CPU time for standard LOO: %.2f seconds\n', time1);

%% LOO decremental approach

svm_train(X_train',Y_train,100,5,1);

tic
loo_est = svmloo();
time2 = toc;

fprintf('Decrementlal LOO error rate: %.4f\n', loo_est/length(X_t));
fprintf('CPU time for decremental LOO: %d\n', time1);
