function result = StatPath(f, y, Elbow)
    yhat = y .* f;
    error = sum(yhat < 0);
    margin = sum(1 - yhat(yhat < 1));
    selbow = length(Elbow);
    result.error = error;
    result.margin = margin;
    result.selbow = selbow;
end