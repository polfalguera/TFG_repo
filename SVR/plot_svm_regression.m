function plot_svm_regression(trnX, trnY, tstX, tstY, ker, C, epsilon)
    % Input Parameters:
    % trnX - Training data features (column vector)
    % trnY - Training data targets (column vector)
    % tstX - Testing data features (column vector)
    % tstY - Testing data targets (column vector)
    % ker - Kernel function (e.g., 'linear', 'rbf')
    % C - Box constraint parameter
    % epsilon - Epsilon margin parameter

    % Fit the SVM regression model
    mdl = fitrsvm(trnX, trnY, 'KernelFunction', ker, 'BoxConstraint', C, 'Epsilon', epsilon);

    % Predict the values for the training and testing data
    predY_trn = predict(mdl, trnX);
    predY_tst = predict(mdl, tstX);

    % Identify support vectors
    sv_indices = mdl.IsSupportVector;
    svX = trnX(sv_indices);
    svY = trnY(sv_indices);

    % Plot the results
    figure;
    hold on;
    % Plot training data points
    plot(trnX, trnY, 'bo', 'DisplayName', 'Training Data');
    % Plot testing data points
    plot(tstX, tstY, 'ro', 'DisplayName', 'Testing Data');
    % Plot support vectors
    plot(svX, svY, 'ko', 'MarkerSize', 10, 'DisplayName', 'Support Vectors');
    % Plot the predicted function for training data
    plot(trnX, predY_trn, 'b-', 'DisplayName', 'SVM Prediction (Training)');
    % Plot the predicted function for testing data
    plot(tstX, predY_tst, 'r-', 'DisplayName', 'SVM Prediction (Testing)');
    % Plot the epsilon tube
    plot(trnX, predY_trn + epsilon, 'k--', 'DisplayName', 'Epsilon Tube');
    plot(trnX, predY_trn - epsilon, 'k--', 'HandleVisibility', 'off');

    % Add labels and legend
    xlabel('X');
    ylabel('Y');
    legend('show');
    title('SVM Regression with Support Vectors and Epsilon Tube');
    hold off;
end
