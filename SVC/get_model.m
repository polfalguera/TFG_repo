function [SVMmodel] = get_model(X, Y)
    model = load('globals.mat');

    ind_sv = sort([model.ind{1}, model.ind{2}]);
    
    SVMmodel = struct();
    SVMmodel.Alphas = model.a;
    SVMmodel.Bias = model.b;

    SVMmodel.SupportVectors = X(ind_sv,:);
    SVMmodel.IsSupportVector = zeros(length(X),1);
    SVMmodel.IsSupportVector(ind_sv) = 1;
    SVMmodel.SupportVectorsAlphas = model.a(ind_sv);
    SVMmodel.SupportVectorsLabels = model.y(ind_sv);
    SVMmodel.BoxConstraints = model.C;
    SVMmodel.Beta = [];

    if model.type == 1
        kernel = 'linear';
        SVMmodel.Beta = zeros(size(SVMmodel.SupportVectors, 2), 1);
        for i = 1:length(SVMmodel.SupportVectorsAlphas)
            SVMmodel.Beta = SVMmodel.Beta + ...
                            SVMmodel.SupportVectorsAlphas(i) * ...
                            SVMmodel.SupportVectorsLabels(i) * ...
                            SVMmodel.SupportVectors(i, :)';
        end
    elseif model.type > 2 && model.type < 5
        kernel = 'polynomial';
    else
        kernel = 'rbf';
    end
    SVMmodel.KernelParameters = struct('Function', kernel, 'Scale', model.scale);
    
end