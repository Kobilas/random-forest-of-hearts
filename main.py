# Matthew Kobilas
# 04/25/20
# Random Forest evaluation program
# should be able to use on other datasets, but for my testing purposes
# I used the UCI Heart Disease dataset

from csv import reader
from random import seed
from random import randrange
from math import sqrt, floor, inf
from operator import itemgetter

# loads csv to list of lists
# prepares it using str_flt_col_to_float for all columns except class value column
# and str_int_col_to_int for class value column
def load_prep_csv(filename):
    dt = []
    with open(fname, 'r') as f:
        csv_rdr = reader(f)
        next(csv_rdr, None) # skip header
        for row in csv_rdr:
            dt.append(row)
    # for each attribute in dataset
    # len(dt[0]) - 1 is length of first list of lists, minus the class variable
    for i in range(0, len(dt[0]) - 1):
        str_flt_col_to_float(dt, i)
    str_int_col_to_int(dt, len(dt[0]) - 1) # converts class value column to enumeration of it in terms of int
    return dt

# convert string in column in dataset (list of lists) to float
# column is the idx of the column in the list of list
# may be more efficient to convert only the columns we know need to be converted to float
def str_flt_col_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip()) # strip whitespace and convert to float

# converts string values in class column to enumeration of integers
def str_int_col_to_int(data, column):
    class_column = [row[column] for row in data] # create list representing column of class values
    unique_classes = set(class_column) # get all unique values of class column
    enum_classes = {}
    for i, x in enumerate(unique_classes):
        enum_classes[x] = i
    for row in data:
        row[column] = enum_classes[row[column]]

# partitions data into num_partitions and returns it as a list of list of lists
def partition_data(data, num_partitions):
    dt_cp = list(data) # copy of data so that records may be randomly indexed and popped
    max_len_part = floor(len(data) / num_partitions) # max length of each partition
    dt_parts = [[] for empty_part in range(num_partitions)]
    for part_i in range(num_partitions):
        while len(dt_parts[part_i]) < max_len_part:
            rand_idx = randrange(len(dt_cp)) # get random record from data
            dt_parts[part_i].append(dt_cp.pop(rand_idx)) # append to current partition we are creating
    return dt_parts

# calculates and returns accuracy as a percentage float
def get_accuracy(actual, prediction):
    num_correct = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            num_correct += 1
    return (num_correct / float(len(actual))) * 100.0

# evaluting random forest using k-fold cross-validation where k is num_partitions
# can add repetition using different seeds to allow for "larger" dataset in the future
def evaluate_en_masse(data, num_partitions, max_tree_depth, min_size, subsample_ratio, num_trees, num_features):
    dt_parts = partition_data(data, num_partitions)
    mass_scores = []
    for part in dt_parts:
        dt_train = dt_parts # right now dt_parts is a list (dataset) of lists (parts) of lists (rows)
        dt_train.remove(part) # remove fold that will be used as test dataset
        dt_train = sum(dt_train, []) # collapse remaining folds into a list (dataset) of lists (rows)
        dt_test = []
        for row in part:
            row_cp = list(row)
            row_cp[-1] = None # hide result in test set from algorithm
            dt_test.append(row_cp)
        predicted_results = build_predict_random_forest(dt_train, dt_test, max_tree_depth, min_size, subsample_ratio, num_trees, num_features)
        actual_results = [row[-1] for row in part]
        accuracy_measure = get_accuracy(actual_results, predicted_results)
        mass_scores.append(accuracy_measure)
    return mass_scores

# bulk of the program takes place in this function or sub-functions
# creates a random_forest based on training, then evaluates testing and returns the predictions
def build_predict_random_forest(training, testing, max_tree_depth, min_size, subsample_ratio, num_trees, num_features):
    forest = []
    for i in range(num_trees):
        sample_training = random_w_replacement_subsample(training, subsample_ratio)
        new_tree = build_tree(sample_training, max_tree_depth, min_size, num_features)
        forest.append(new_tree)
    predictions = [predict_w_bagging(forest, row) for row in testing]
    return predictions

# returns ratio size of data
# samples with replacement, so the original sample is not decreased between selections
def random_w_replacement_subsample(data, ratio):
    dt_sub = []
    subsample_size = round(len(data) * ratio) # rounds to the nearest int
    while len(dt_sub) < subsample_size:
        rand_idx = randrange(len(data))
        dt_sub.append(data[rand_idx]) # appends random record to subsample, but does not pop (selection with replacement)
    return dt_sub

# creates a single decision tree
def build_tree(training, max_tree_depth, min_size, num_features):
    root = get_branch(training, num_features)
    split_or_terminate(root, max_tree_depth, min_size, num_features, 1) # current depth is 1 because we just created root
    return root

# discerns branching point from data based on number of features indicated
# gets random set of features (columns) each time
def get_branch(data, num_features):
    unique_classes = list(set(row[-1] for row in data)) # get list of unique class values
    #branch_idx, branch_val, branch_score, branch_branches = inf, inf, inf, None
    branch_idx, branch_val, branch_score, branch_branches = 999, 999, 999, None
    ls_features = []
    # loop for num_features, and randomly add column indices to ls_features
    # no error checking, may loop forever if number of features is greater than number of columns in dataset
    while len(ls_features) < num_features:
        rand_idx = randrange(len(data[0])-1) # get random column idx that isn't class column
        if rand_idx not in ls_features:
            ls_features.append(rand_idx)
    # need to sort either here, in gini_presplit, or in gini_value
    # will split in gini_presplit
    for idx_feat in ls_features:
        for row in data:
            branches = gini_presplit(data, idx_feat, row[idx_feat]) # splits data into tuple of (left, right) based on value of row[idx_feat]
            gini = gini_value(branches, unique_classes) # calculate gini values, given tuple of split lists and the unique class values found above
            # if resulting gini is better than what we had previously, 
            # change all the values we have gotten so far to the newly determined ones
            if gini < branch_score:
                branch_idx = idx_feat
                branch_val = row[idx_feat]
                branch_score = gini
                branch_branches = branches
    # return dictionary of which branching attribute to use, the value that caused us to get the minimal gini
    # as well as the tuple of the split that we used to find it
    # gini score also returned for debugging purposes
    return {"index": branch_idx,
            "value": branch_val,
            "score": branch_score,
            "branches": branch_branches}

# splits data into left and right halves based on the attr_index column, and the value in attr_value
def gini_presplit(data, attr_index, attr_value):
    l = []
    r = []
    # sort the data, since data must be sorted prior to calculating gini values
    # not necessarily necessary considering we are looping through all rows in get_branch() anyway
    # if code does not work accurately, it may arise from index errors due to this sort
    dt_sort = sorted(data, key=itemgetter(attr_index))
    for row in dt_sort:
        if row[attr_index] < attr_value:
            l.append(row)
        else:
            r.append(row)
    return (l, r)

# calculate gini value with 2-length tuple of split data
# along with unique class values
def gini_value(data_split, unique_class_values):
    num_records = float(sum([len(splt) for splt in data_split])) # count total number of records among all splits
    gini = 0.0 # initialize with purest possible Gini
    for splt in data_split:
        splt_len = float(len(splt))
        if splt_len == 0: # size of split can be zero if the node is completely pure
            continue
        pre_score = 0.0
        for value in unique_class_values: # scoring split on each unique class value in dataset
            pre_score += ([row[-1] for row in splt].count(value) / splt_len)**2 # determining branch score without weight first
        gini += (1.0 - pre_score) * (splt_len / num_records)
    return gini

def split_or_terminate(node, max_tree_depth, min_size, num_features, depth_of_node):
    l, r = node["branches"]
    del(node["branches"])
    if not l or not r: # checking for false branch
        node["l"] = node["r"] = terminate(l + r)
        return
    if depth_of_node >= max_tree_depth: # checking if this node hit max depth
        node["l"], node["r"] = terminate(l), terminate(r)
        return
    if len(l) <= min_size: # checking if left child is less than min size and whether we need to process it further
        node["l"] = terminate(l)
    else: # if it is greater than min size, we get branching attribute and call split_or_terminate recursively
        node["l"] = get_branch(l, num_features)
        split_or_terminate(node["l"], max_tree_depth, min_size, num_features, depth_of_node+1)
    if len(r) <= min_size: # checking if right child is less than min size and whether we need to process further
        node["r"] = terminate(r)
    else: # if it is greater than min size, we get branching attribute and call split_or_terminate recursively
        node["r"] = get_branch(r, num_features)
        split_or_terminate(node["r"], max_tree_depth, min_size, num_features, depth_of_node+1)

# create terminal node branch in tree, meaning we have achieved pure node
# or that gini index is low enough where we do not have to split any further
def terminate(branch):
    resulting_class_values = [row[-1] for row in branch]
    return max(set(resulting_class_values), key=resulting_class_values.count)

# make a prediction using bagging and a forest on a single record in dataset
def predict_w_bagging(forest, record):
    results = [predict_w_tree(tree, record) for tree in forest] # predict what result of record is for each tree in the forest
    # return the mode of the results of predicting with each tree in the forest
    # this is akin to voting, as is usual in ensemble learning
    return max(set(results), key=results.count)

# classify a single record using a single tree
def predict_w_tree(node, record):
    # check if the value at branching attribute is less than branching condition
    if record[node["index"]] < node["value"]:
        # if it is, then we check if the node at that branch is a dictionary
        if isinstance(node["l"], dict):
            # if its a dictionary, we call this function recursively to continue classifying
            return predict_w_tree(node["l"], record)
        # or not
        else:
            # if its not, then we just return the classification found there
            return node["l"]
    # if its not less than the branching condition, we continue down the right branch instead
    else:
        # if the node at end of branch is a dictionary
        if isinstance(node["r"], dict):
            # then we call this function recursively and continue classfiying
            return predict_w_tree(node["r"], record)
        else:
            # otherwise we return the classification
            return node["r"]

seed(666) # set random seed, 666 for my ucid mk666
# load data to list and prep it
fname = "heart.csv"
data = load_prep_csv(fname)
# good value for k is 5, making training set be 80% and testing set be 20%
k = 5
max_dep = 10
min_sz = 1
sample_rate = 1.0
# num_features best results are either sqrt or log_2 according to google
num_features = int(sqrt(len(data[0])-1))
# loop through num_trees eventually
num_trees = [1, 5, 10]
for i in num_trees:
    results = evaluate_en_masse(data, k, max_dep, min_sz, sample_rate, i, num_features)
    print("Trees: " + str(i))
    print("Scores: " + str(results))
    print("Average Accuracy: %.3f%%" % (sum(results) / float(len(results))))