% DECISION_TREE_CLASSIFIER_MAIN Main program to test the functioning of the
% decision tree. Generates binary dataset and replaces two columns with
% numerical values. Also generates corresponding labels.
% Trains the tree and does a test classification.

% Generating dataset
ds = flip(de2bi([0:7]'),2);
% Generating corresponding labels
labels = (ds(:,1) | ds(:,2)) & ds(:,3);
% Replacing first and last column with numerical values
ds(:,end) = [10; 70; 20; 80; 40; 60; 50; 60];
ds(:,1) = [30; 30; 30; 30; 60; 60; 60; 60];
% Vector to classify
x = [90 1 10];
% Labeling numerical columns
numerical_cols = [1 3];
% Printing the dataset
fprintf('Dataset -> Labels\n');
fprintf('%d %d %d -> %d\n',[ds'; labels']);
fprintf('\nVector to classify: x = [%d %d %d]\n',x);
% Building the tree and classifying
fprintf('Training the decision tree...\n');
[fv, c, thresholds] = construct_tree(ds, labels, [], [], 1, [], mode(labels), [], numerical_cols, []);
out = classifier(fv, c, x, numerical_cols, thresholds);

% Printing result
fprintf('Assigned label = %d\n', out);