# ID3 Decision Tree algorithm
A MATLAB implementation of the [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) decision tree building algorithm for classification tasks.  
The algorithm was extended to include the handling of numerical attributes, as in the [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm) evolution of the original ID3 algorithm. However, this implementation does not include features such as a-posteriori pruning.

## MATLAB scripts
- `decision_tree_classifier_main.m` The main script to launch the program
- `classifier.m` Performs the classification task given a tree representation and a sample to classify. Returns the label predicted for the given sample
- `construct_tree.m` Recursively construct a decision tree classifier based on a given dataset and a set of labels
- `preprocess_numerical.m` Preprocess a numerical feature to adapt it to the ID3 algorithm
- `get_prob_vect.m` and `H.m` are simple utility functions to create probability vectors out of a set of data and to compute the entropy of an array, respectively

## Input/output format
Training data and labels should be provided to `construct_tree` as a *N&ast;m* and *m* sized matrix and array respectively. Labels should be in integer format.  
Predicted label is returned as a single integer.
