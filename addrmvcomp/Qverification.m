% Get random values
clear; clc;

Q_prime = [10, 5, 3, 6; 5, 15, 6, 2; 3, 6, 20, 12; 4, 9, 3, 7]; 
Q_prime_inv = inv(Q_prime);
u = [2; 3; 1; 5];
U = u * u';
v = [5; 6; 7; 8];
delta_p = 0.2;
alphaS = [1;2;3];

up = delta_p * Q_prime_inv * (u .* u') * Q_prime_inv;
down = 1 + delta_p * u' * Q_prime_inv * u;
Q_prime_inv = Q_prime_inv - (up / down);

increments = (delta_p * Q_prime_inv) * (-U * [0;alphaS] - v);

[delta_alphaS, delta_bias] = solve_equation(Q_prime,U,v,alphaS,delta_p);

% Enough round to ensure no differences in outcome.
if round(increments,10) == round([delta_bias;delta_alphaS],10)
    disp('Equal results.')
else
    disp('Different outcomes.')
end