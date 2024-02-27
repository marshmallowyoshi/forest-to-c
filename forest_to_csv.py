"""
NAME
    forest_to_csv - Stores information required to recreate a RandomForestClassifier in '.csv' and
    '.meta' files. They are formatted to be read by the csv_to_c module.

FUNCTIONS
    tree_to_df - Extracts tree data to a Pandas DataFrame.
    forest_to_df - Extracts forest data to a Pandas DataFrame.
    add_branches - Adds index of branches to the tree data.
    write_metadata - Writes metadata to a '.meta' file.
    forest_struct_to_csv - Generates '.csv' and '.meta' files from a RandomForestClassifier.
"""
import pandas as pd

def tree_to_df(tree):
    """
    Convert a tree model to a DataFrame.

    Parameters:
    tree (object): The tree model to be converted.

    Returns:
    DataFrame: A DataFrame containing the tree model information.
    """
    tree_ = tree.tree_
    tree_df = pd.DataFrame(columns=['depth', 'threshold', 'feature', 'value'])
    depths = tree_.compute_node_depths()
    for idx, val in enumerate(tree_.threshold):
        if tree_.children_left[idx] == -1:
            value = [int(x) for x in tree_.value[idx][0]]
            val = pd.NA
            feature = pd.NA
        else:
            value = pd.NA
            feature = tree_.feature[idx]
        tree_df.loc[idx] = [depths[idx], val, feature, value]
    return tree_df

def forest_to_df(forest):
    """
    Convert a forest of trees to a single DataFrame.

    Args:
        forest: The forest of trees to be converted to a DataFrame.

    Returns:
        DataFrame: A single DataFrame containing the combined results of all trees in the forest.
    """
    return pd.concat([tree_to_df(tree) for tree in forest.estimators_], ignore_index=True)

def add_branches(tree):
    """
    Add branches location indexes to the tree DataFrame based on the depth of each node.
    Parameters:
    - tree: the tree DataFrame to which branches will be added
    
    Returns:
    - The tree DataFrame with branches added
    """
    tree['branches'] = [pd.NA for x in range(len(tree))]
    for idx, row in enumerate(tree.itertuples()):
        idx += tree.index[0]
        branch_list = [x
                       for x in tree[tree['depth'] == row.depth + 1].index.to_list()
                       if x > idx][:2]

        if len(branch_list) == 0:
            continue
        elif branch_list[0] != idx+1:
            continue
        else:
            tree.at[idx, 'branches'] = branch_list

    return tree

def write_metadata(rf, file_name):
    """
    Write metadata to a file.

    Args:
        rf: The random forest model.
        file_name: The name of the file to write the metadata to.

    Returns:
        None
    """
    largest_sample_size = int(max([max(rf.estimators_[x].tree_.n_node_samples)
                                   for x in range(len(rf.estimators_))]))
    feature_count = rf.n_features_in_
    tree_count = len(rf.estimators_)
    class_count = len(rf.classes_)
    max_depth = rf.max_depth if rf.max_depth is not None else 32767
    classes = rf.classes_

    with open(file_name, 'w', encoding='utf-8-sig') as f:
        f.write(f'largest_sample_size: {largest_sample_size}\n')
        f.write(f'feature_count: {feature_count}\n')
        f.write(f'tree_count: {tree_count}\n')
        f.write(f'class_count: {class_count}\n')
        f.write(f'max_depth: {max_depth}\n')
        f.write('classes: ')
        for c in classes:
            f.write(f'{c}, ')
    return None

def forest_struct_to_csv(rf, file_name):
    """
    Converts the structure of a given random forest model to a CSV file. 

    Args:
        rf: The random forest model to convert.
        file_name: The name of the CSV file to be created.

    Returns:
        pandas.DataFrame: The converted forest structure as a DataFrame.
    """
    tree = rf.estimators_[0]
    tree_ = tree.tree_
    thresholds = tree_.threshold
    tree_df = pd.DataFrame(columns=['depth', 'threshold', 'feature', 'is_leaf', 'value'])
    depths = tree_.compute_node_depths()
    for idx, val in enumerate(thresholds):
        tree_df.loc[idx] = [depths[idx],
                            val,
                            tree_.feature[idx],
                            tree_.children_left[idx] == -1,
                            tree_.value[idx][0]]

    forest = forest_to_df(rf)

    tree_start = forest[forest['depth']==1].index.to_list()
    tree_list = []

    for idx, val in enumerate(tree_start):
        if idx == len(tree_start) - 1:
            tree_list.append(forest.loc[val:])
        else:
            tree_list.append(forest.loc[val:tree_start[idx+1]-1])

    new_tree_list = [add_branches(x) for x in tree_list]
    new_forest = pd.concat(new_tree_list, ignore_index=True)
    new_forest.to_csv("".join((file_name, '.csv')), index=False)
    write_metadata(rf, "".join((file_name, '.meta')))
    return new_forest
