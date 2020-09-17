function [fv, c, thresholds] = construct_tree(ds, labels, fv, c, k, path, common_c_parent, used_cols, numerical_cols, thresholds)
% CONSTRUCT_TREE Recursively construct a decision tree classifier based on a given
% dataset and a set of numerical labels.
%   [fv, c, thresholds] = CONSTRUCT_TREE(ds, labels, fv, c, k, path, common_c_parent, used_cols, numerical_cols, thresholds)
%   The function takes the following parameters:
%   - ds: dataset of size RxC. All values must be either numerical values or labels in
%   numerical form.
%   - labels: R-sized column vector of numerical class labels.
%   - fv: data structure representing the tree. Memorizes in order the
%   (feature, value) couples of all possible root-leaf paths. When
%   classifying a vector, exact correspondance between its (feature, value)
%   couples and a tree path is sought. By design, a vector must end up in
%   some leaf of the tree. The case in which not all of the features are
%   considered for classification is also handled by the program.
%   - c: possible classification labels. Has the same number of rows as fv
%   (one for each possible classification outcome)
%   - k: recursion depth
%   - common_c_parent: most common class in parent node (used when
%   backtracking from leaf with empty dataset)
%   - used_cols: keeps track of what columns have been considered for the
%   split
%   - numerical_cols: indicates which columns of the dataset must be
%   considered numerical
%   - thresholds: saves split threshold values for each numerical attribute
%   Returns fv representation of the tree, c labels and thresholds to be used by the
%   classifier function.

p = get_prob_vect(labels);
H_c = H(p);
all_cols = 1:size(ds,2);
% Stop condition: if current label entropy is 0 or all columns have already
% been used
if H_c == 0 || isempty(ds(:,setdiff(all_cols, used_cols)))
    % If the dataset is not empty, then all labels are the same. We just
    % assign the first one.
    if ~isempty(labels)
        c = [c; labels(1)];
    else
        % Otherwise, we assign the most common parent label
        c = common_c_parent;
    end
    % Appending leaf path to fv (matrix of root-leaf paths). We first resize
    % fv or path as needed (i.e. if one is bigger than the other, we make
    % them equal)
    if length(path)~=size(fv,2) && ~isempty(fv)
        if size(fv,2) > length(path)
            % struct vectors are resized just by adding empty values to the
            % desired position of the vector
            path(size(fv,2)).f = [];
        else
            fv(length(path)).f = [];
        end
    end
    fv = [fv; path];
    return
end

% We compute the most common class label of the current portion of the
% dataset. This will be used by the child node, if a stop condition with
% an empty dataset occurs.
common_c = mode(labels);

% Compute conditional entropies and IGR for each feature
IGR = zeros(1,size(ds,2));
% Cycle through all columns that have not been used yet
temp_ds = ds;
for j = setdiff(all_cols, used_cols)
    % check if the current column is numerical. If so, we replace it with
    % it binary version split in the correct threshold
    [is_numerical, numerical_col_index] = ismember(j, numerical_cols);
    if is_numerical
        [temp_ds(:,j), best_t] = preprocess_numerical(ds(:,j), labels);
        thresholds(numerical_col_index) = best_t;
    end
    p_xj = get_prob_vect(temp_ds(:,j));
    H_xj = H(p_xj);
    H_cx = zeros(1, length(p_xj));
    % Computing H(C|X=xj) by looping through all possible values of p_xj.
    % Values are implicitly considered as indices
    for i=1:length(p_xj)
        labels_filtered = labels(temp_ds(:,j) == i,:);
        H_cx(i) = H(get_prob_vect(labels_filtered));
    end
    H_cxj = sum(p_xj.*H_cx);
    I_cxj = H_c - H_cxj;
    IGR(j) = I_cxj/H_xj;
end

% Split on max IGR feature
[~, best_feat] = max(IGR);
used_cols = [used_cols best_feat];

% Split rows on best feature values
for val = unique(temp_ds(:,best_feat))'
    % Select the records in which the selected feature has value val
    sub_ds = ds(temp_ds(:,best_feat) == val, :);
    % Do the same for labels
    sub_labels = labels(temp_ds(:,best_feat) == val);
    
    % Create a new step of the path
    step.v = val;
    step.f = best_feat;
    % Recursive call: increase depth of the k counter
    [fv, c, thresholds] = construct_tree(sub_ds, sub_labels, fv, c, k+1, [path step], common_c, used_cols, numerical_cols, thresholds);
end

end