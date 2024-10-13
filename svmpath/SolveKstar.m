function bstar = SolveKstar(Kstar, ridge)
    if nargin < 2
        ridge = 1e-10;
    end
    
    onestar = ones(1, size(Kstar, 2));
    onestar(1) = 0;

    if ridge > 0
        Kstar = Kstar + ridge * eye(length(onestar));
    end
    
    bstar = (Kstar \ onestar')';
end