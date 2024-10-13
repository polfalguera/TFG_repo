function [new_Q_prime] = contractQ(Q_prime_inv,removed_inds)
    
    for k = removed_inds' + 1
        L_S = size(Q_prime_inv, 1);
        for i = 1:L_S
            for j = 1:L_S
                if (i ~= k && j ~= k)
                    Q_prime_inv(i,j) = Q_prime_inv(i,j) - ...
                        ( (Q_prime_inv(i,k) * Q_prime_inv(k,j)) / (Q_prime_inv(k,k)) );
                end
            end
        end
        Q_prime_inv(k,:) = [];
        Q_prime_inv(:,k) = [];
    end
    
    new_Q_prime = inv(Q_prime_inv);
end