function [new_Q_prime] = expandQ(X,y,Q_prime_inv,feature,old_inds,new_inds,p)

    added_inds = setdiff(new_inds, old_inds);
    
    for j = added_inds'
        Q_prime_plus = [y(j)];
        idx = 1;
        for i = old_inds'
            idx = idx+1;
            Q_prime_plus(idx) = y(i) * y(j) * (dot(X(i,:),X(j,:)') + p * feature(i) * feature(j));
        end

        Q_prime_plus = Q_prime_plus';
        beta = -Q_prime_inv * Q_prime_plus;
        Q_prime_added_point = y(j) * y(j) * (dot(X(j,:),X(j,:)') + p * feature(j) * feature(j));
        gamma =  Q_prime_added_point - Q_prime_plus' * Q_prime_inv * Q_prime_plus;

        temp = [Q_prime_inv;zeros(1,size(Q_prime_inv,1))];
        temp = [temp,zeros(size(temp,1),1)];

        Q_prime_inv = temp + (1 / gamma) * [beta;1] * [beta',1];
        
        pos = find(new_inds == j) + 1;
        rows = [Q_prime_inv(1:pos-1,:); Q_prime_inv(end,:); Q_prime_inv(pos:end-1,:)];
        Q_prime_inv = [rows(:,1:pos-1),rows(:,end), rows(:,pos:end-1)];

        old_inds = [old_inds;j];
    end
    
        
    new_Q_prime = inv( Q_prime_inv );
    new_Q_prime(:,1) = round(new_Q_prime(:,1));
    new_Q_prime(1,:) = round(new_Q_prime(1,:));
end