function p = get_prob_vect(X)
% GET_PROB_VECT(X) compute a probability vector out of X. Counts the
% frequency of occurrence of each value and divides it by the number of
% total elements. This is interpreted as a probability.
un_X = unique(X);
p = zeros(1, length(un_X));

for i=1:length(p)
    p(i) = sum(X == un_X(i))/length(X);
end

end