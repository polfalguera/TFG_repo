
clear X Y; clc;

[X, Y] = dsread_adult();

X_train = X(1:5000,:);
Y_train = Y(1:5000);

% TRAIN fitcsvm

tic
svmModel = fitcsvm(X_train,Y_train,'BoxConstraint',100,'KernelFunction','rbf','KernelScale',"auto",'Solver','L1QP');
time1 = toc;

fprintf('CPU time for fitcsvm: %d\n', time1);

% TRAIN svm_train
X_train = X(1:5010,:);
Y_train = Y(1:5010);

tic
svm_train(X_train',Y_train,100,5,power(svmModel.KernelParameters.Scale,2));
time2 = toc;

mysvmModel = get_model(X_train,Y_train);

fprintf('CPU time for svm_train: %d\n', time2);

%% STATS

X_test = X(5002:6001,:);
Y_test = Y(5002:6001,:);
disp('Stats for fitcsvm:');
fprintf('Number of Support Vectors: %d\n', length(find(svmModel.IsSupportVector)));

pl1 = svmModel.predict(X_test);
acc1 = sum(Y_test == pl1)/length(Y_test);
fprintf('Accuracy (%%): %d\n', acc1*100);
error1 = 1 - acc1;
fprintf('Error (%%): %d\n\n', error1 *100);

disp('Stats for svm_train:');
fprintf('Number of Support Vectors: %d\n', length(find(mysvmModel.IsSupportVector)));

pl2 = sign(svmeval(X_test'));
acc2 = sum(Y_test == pl2)/length(Y_test);
fprintf('Accuracy (%%): %d\n', acc2*100);
error2 = 1 - acc2;
fprintf('Error (%%): %d\n', error2*100);