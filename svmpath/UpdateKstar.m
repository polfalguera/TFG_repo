function Kstar = UpdateKstar(Kstar, Kell, Krow, y)
    if length(y) == 1
        advec = [y, Krow];
        Kstar = [Kstar; advec];
        Kstar = [Kstar, [advec, Kell]'];
    else
        adrect = [y, Krow];
        Kstar = [Kstar; adrect];
        Kstar = [Kstar, [adrect'; Kell]];
    end
end