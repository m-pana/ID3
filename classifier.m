function label = classifier(fv, c, x, numerical_cols, thresholds)
% CLASSIFIER take a tree representation fv and a list of labels c. Return
% the corresponding class label.
%   label = CLASSIFIER(fv, c, x, numerical_cols, thresholds) takes the
%   following arguments:
%   - fv: matrix representation of the decision tree generated by
%   construct_tree
%   - c: classification labels generated by construct_tree
%   - x: vector to classify
%   - numerical_cols: indicates which columns are numerical (to convert
%   them to binary labels)
%   - thresholds: indicates threshold on which to split numerical labels
%   Returns numerical classification label

% Initializing label variable
label = -1;
% First converting numerical cols into their respective values of super
% awesomeness
cols_to_convert = x(numerical_cols);
cols_to_convert(cols_to_convert <= thresholds) = 0;
cols_to_convert(cols_to_convert > thresholds) = 1;
x(numerical_cols) = cols_to_convert;

% Unwrap the fv structure row by row in two arrays: one that indicates the
% features to use, one that contains each corresponging value
for i = 1:size(fv,1)
    [f,v] = split_fv(fv(i,:));
    % Verify if all features of the vector match the path. If so, classify
    % it accordingly and break the loop
    if all(x(f) == v)
        label = c(i);
        break
    end
end
end


function [f, v] = split_fv(fv_row)
% SPLIT_FV utility function to split a row of the fv structure into two
% separate f and v arrays.
%   [f,v] = SPLIT_FV(fv_row) return the fv_row given in two separate arrays
%   f and v.
for i=1:length(fv_row)
    % If the structure has empty spaces (paddings) then stop the split
    % process
    if ~isempty(fv_row(i).f)
        f(i) = fv_row(i).f;
        v(i) = fv_row(i).v;
    end
end
end