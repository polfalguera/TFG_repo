function Kstar = DowndateKstar(Kstar, index)
    index = index + 1;
    Kstar(index, :) = [];
    Kstar(:, index) = [];
end