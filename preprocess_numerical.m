function [new_values, best_t] = preprocess_numerical(values, labels)
H_c = H(get_prob_vect(labels));
%IGR vector stores individual IGRs
uniq_vals = unique(values);
IGRs = [];
for t = uniq_vals'
    % Map into binary values
    new_values = values;
    new_values(new_values <= t) = 0;
    new_values(new_values > t) = 1;
    % Entropy of X
    px = get_prob_vect(new_values);
    H_x = H(px);
    % Conditional entropy of C given X=0, i.e. H(C|X=0)
    H_cx0 = H(get_prob_vect(labels(new_values == 0)));
    % Conditional entropy of C given X=0, i.e. H(C|X=1)
    H_cx1 = H(get_prob_vect(labels(new_values == 1)));
    % Storing them in a vector for ease of multiplication...
    % Total conditional entropy of H(C|X)
    H_cX = sum(px.*[H_cx0 H_cx1]);
    I_t = H_c - H_cX;
    IGR_t = I_t/H_x;
    % Appending the new IGR_t value in a IGR vector
    IGRs = [IGRs IGR_t];
end

% Selecting the maximum IGR and the best t
[max_IGR, max_ind] = max(IGRs);
best_t = uniq_vals(max_ind);
% Converting the new column
new_values = values;
new_values(new_values <= best_t) = 0;
new_values(new_values > best_t) = 1;
end