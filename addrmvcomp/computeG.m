function [g_i] = computeG(X,y,new_alpha,new_bias,feature,p,i)

    sum = 0;
    epsilon=0.00000000000001;
    L = size(X,1);
    for j = 1:L
        sum = sum + ( y(j) * new_alpha(j) * ( dot(X(i,:),X(j,:)') + p * feature(i) * feature(j) ));
    end
    
    g_i = y(i) * (sum + new_bias) - 1;
    
end